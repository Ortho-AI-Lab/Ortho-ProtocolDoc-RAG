from pdf2image import convert_from_path
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError,
)
from pdfminer.high_level import extract_text
import base64
import io
import os
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
from rich import print
from ast import literal_eval
from pathlib import Path
import dotenv
from functools import partial
import time


file_stems = [
    "Depuy_distal",
    "OSSZimmer_Distal",
    "stryker_cut",
]

stem_to_company = {
    "Depuy_distal": "DePuy",
    "onko_distal": "Onkos",
    "OSSZimmer_Distal": "Zimmer Biomet",
    "stryker_cut": "Stryker",
}

company_to_document_title = {
    "DePuy": "Orthogenesis Limb Preservation System Surgical Techniques",
    "Onkos": "Distal Femoral Replacement Surgical Technique: Passive Fixed Hinge Tibia Option",
    "Zimmer Biomet": "Segmental Distal Femoral Replacement Surgical Technique",
    "Stryker": "GMRS Distal Femur and Revision Baseplate Surgical Protocol",
}


def read_questions(questions_path: Path) -> list[str]:
    """Reads questions from a file and returns a list of questions."""
    with open(questions_path, "r") as f:
        questions = f.readlines()
    questions = [question.strip() for question in questions]
    print("Read", len(questions), "questions from", questions_path)
    return questions


curr_dir = Path(__file__).resolve().parent
dotenv.load_dotenv(curr_dir.parent / ".env")
dir_path = curr_dir.parent
analysis_path = dir_path / "analysis"
storage_path = dir_path / "storage" / "openai"
storage_path.mkdir(parents=True, exist_ok=True)
json_path = storage_path / "openai_jsons.json"
if not json_path.exists():
    json_path.touch()
embed_path = storage_path / "openai_embeddings.csv"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def convert_doc_to_images(path):
    images = convert_from_path(path)
    return images


def extract_text_from_doc(path):
    text = extract_text(path)
    return text


def get_img_uri(img):
    png_buffer = io.BytesIO()
    img.save(png_buffer, format="PNG")
    png_buffer.seek(0)

    base64_png = base64.b64encode(png_buffer.read()).decode("utf-8")

    data_uri = f"data:image/png;base64,{base64_png}"
    return data_uri


def analyze_image(data_uri, system_prompt=""):
    time.sleep(10.0)
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": f"{data_uri}"}}],
            },
        ],
        max_tokens=500,
        temperature=0,
        top_p=0.1,
    )
    time.sleep(10.0)
    return response.choices[0].message.content


def analyze_doc_image(img, system_prompt=""):
    img_uri = get_img_uri(img)
    data = analyze_image(img_uri, system_prompt)
    return data


def get_embeddings(text, embeddings_model="text-embedding-3-large"):
    embeddings = client.embeddings.create(
        model=embeddings_model, input=text, encoding_format="float"
    )
    return embeddings.data[0].embedding


def search_content(df, input_text, top_k):
    embedded_value = get_embeddings(input_text)
    df["similarity"] = df.embeddings.apply(
        lambda x: cosine_similarity(
            np.array(x).reshape(1, -1), np.array(embedded_value).reshape(1, -1)
        )
    )
    res = df.sort_values("similarity", ascending=False).head(top_k)
    return res


def get_similarity(row):
    similarity_score = row["similarity"]
    if isinstance(similarity_score, np.ndarray):
        similarity_score = similarity_score[0][0]
    return similarity_score


def generate_output(
    input_prompt, similar_content, threshold=0.5, system_prompt="", model="gpt-4o"
):
    content = similar_content.iloc[0]["content"]

    # Adding more matching content if the similarity is above threshold
    if len(similar_content) > 1:
        for i, row in similar_content.iterrows():
            similarity_score = get_similarity(row)
            if similarity_score > threshold:
                content += f"\n\n{row['content']}"

    prompt = f"INPUT PROMPT:\n{input_prompt}\n-------\nCONTENT:\n{content}"

    completion = client.chat.completions.create(
        model=model,
        temperature=0.5,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    return completion.choices[0].message.content


def populate_jsons():
    file_paths = [
        dir_path / "agents_llamaindex" / "reference_docs" / f"{filestem}.pdf"
        for filestem in file_stems
    ]
    docs = []

    for path in file_paths:
        time.sleep(10.0)
        doc = {"filename": path.name}
        text = extract_text_from_doc(path)
        doc["text"] = text
        imgs = convert_doc_to_images(path)
        pages_description = []

        print(f"Analyzing pages for doc {path.name}")

        system_prompt = """\
You will be provided with an image of a PDF page or a slide. Your goal is to deliver a detailed and engaging presentation about the content you see, using clear and accessible language suitable for a 101-level audience.

If there is an identifiable title, start by stating the title to provide context for your audience.

Describe visual elements in detail:

- **Diagrams**: Explain each component and how they interact. For example, "The process begins with X, which then leads to Y and results in Z."

- **Tables**: Break down the information logically. For instance, "Product A costs X dollars, while Product B is priced at Y dollars."

Focus on the content itself rather than the format:

- **DO NOT** include terms referring to the content format.

- **DO NOT** mention the content type. Instead, directly discuss the information presented.

Keep your explanation comprehensive yet concise:

- Be exhaustive in describing the content, as your audience cannot see the image.

- Exclude irrelevant details such as page numbers or the position of elements on the image.

Use clear and accessible language:

- Explain technical terms or concepts in simple language appropriate for a 101-level audience.

Engage with the content:

- Interpret and analyze the information where appropriate, offering insights to help the audience understand its significance.

------

If there is an identifiable title, present the output in the following format:

{TITLE}

{Content description}

If there is no clear title, simply provide the content description.
        """

        # Concurrent execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(
                    partial(analyze_doc_image, system_prompt=system_prompt), img
                )
                for img in imgs
            ]

            with tqdm(total=len(imgs) - 1) as pbar:
                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(1)

            for f in futures:
                res = f.result()
                pages_description.append(res)

        doc["pages_description"] = pages_description
        docs.append(doc)

    print(len(docs))

    with open(json_path, "w") as f:
        json.dump(docs, f)

    print(f"Saved JSONs to {json_path}\n\n")
    print("\n\nJSONs populated.")


def embed_content():
    print(f"JSON Size: {os.path.getsize(json_path)}")
    with open(json_path, "r") as f:
        print(f"JSON Size: {os.path.getsize(json_path)}")
        docs = json.load(f)

    content = []
    for doc in docs:
        print(f"Obtaining embeddings for doc {doc['filename']}")

        text = doc["text"].split("\f")
        description = doc["pages_description"]
        description_indexes = []
        for i in range(len(text)):
            slide_content = text[i] + "\n"
            # Trying to find matching slide description
            slide_title = text[i].split("\n")[0]
            for j in range(len(description)):
                description_title = description[j].split("\n")[0]
                if slide_title.lower() == description_title.lower():
                    slide_content += description[j].replace(description_title, "")
                    # Keeping track of the descriptions added
                    description_indexes.append(j)
            # Adding the slide content + matching slide description to the content pieces
            content.append(slide_content)
        # Adding the slides descriptions that weren't used
        for j in range(len(description)):
            if j not in description_indexes:
                content.append(description[j])

    clean_content = []
    for c in content:
        text = (
            c.replace(" \n", "").replace("\n\n", "\n").replace("\n\n\n", "\n").strip()
        )
        text = re.sub(r"(?<=\n)\d{1,2}", "", text)
        text = re.sub(r"\b(?:the|this)\s*slide\s*\w+\b", "", text, flags=re.IGNORECASE)
        clean_content.append(text)

    df = pd.DataFrame(clean_content, columns=["content"])

    df["embeddings"] = df["content"].apply(
        lambda x: get_embeddings(x, embeddings_model="text-embedding-3-large")
    )

    print(f"Saving embeddings to {embed_path}\n\n")

    print(df.head())

    print("\n\nEmbedding complete.")

    df.to_csv(embed_path, index=False)


def rag():
    df = pd.read_csv(embed_path)
    df["embeddings"] = df.embeddings.apply(literal_eval).apply(np.array)

    system_prompt = """
You will be provided with an input prompt and content as context that can be used to reply to the prompt.

You will do 2 things:

1. First, you will internally assess whether the content provided is relevant to reply to the input prompt. 

2a. If that is the case, answer directly using this content. If the content is relevant, use elements found in the content to craft a reply to the input prompt.

2b. If the content is not relevant, use your own knowledge to reply or say that you don't know how to respond if your knowledge is not sufficient to answer.

Stay concise with your answer, replying specifically to the input prompt without mentioning additional information provided in the context content.
"""

    model = "gpt-4o"

    for file_stem in file_stems:
        company_name = stem_to_company[file_stem]
        document_title = company_to_document_title[company_name]

        document_title_wo_spaces = document_title.replace(" ", "_")

        results_dir = analysis_path / "results_openai" / document_title_wo_spaces
        results_dir.mkdir(parents=True, exist_ok=True)

        question_prefix = f"Based on the document titled {document_title} from {company_name}, answer this question:\n\n"

        questions = read_questions(analysis_path / "questions.txt")
        questions = [question_prefix + question for question in questions]

        answers = []

        for question in questions:
            ex = question
            print(f"[deep_pink4][bold]QUERY:[/bold] {ex}[/deep_pink4]\n\n")
            matching_content = search_content(df, ex, 3)
            print(f"[grey37][b]Matching content:[/b][/grey37]\n")
            for i, match in matching_content.iterrows():
                print(
                    f"[grey37][i]Similarity: {get_similarity(match):.2f}[/i][/grey37]"
                )
                print(
                    f"[grey37]{match['content'][:100]}{'...' if len(match['content']) > 100 else ''}[/[grey37]]\n\n"
                )
            reply = "test"
            reply = generate_output(
                ex, matching_content, system_prompt=system_prompt, model=model
            )
            print(
                f"[turquoise4][b]REPLY:[/b][/turquoise4]\n\n[spring_green4]{reply}[/spring_green4]\n\n--------------\n\n"
            )

            time.sleep(2.0)

            answers.append(reply)

        with open(results_dir / "openai_answers.txt", "w") as f:
            for i, question in enumerate(questions):
                f.write("-----------------\n")
                f.write(f"Question: {question}\n")
                f.write(f"Answer: {answers[i]}\n\n\n")


if __name__ == "__main__":
    # populate_jsons()
    # embed_content()
    rag()
