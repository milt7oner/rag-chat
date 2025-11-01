# services/rag_service.py
from openai import OpenAI
from schema.rag_schema import InputMessage
import numpy as np

client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)

# 🔹 Cargar archivo base
def load_knowledge(file_path: str = "data/knowledge.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# 🔹 Crear embedding para un texto
def get_embedding(text: str):
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(embedding.data[0].embedding)

# 🔹 Recuperar contexto relevante
def retrieve_context(query: str, knowledge_text: str, top_k: int = 3):
    paragraphs = [p.strip() for p in knowledge_text.split("\n") if p.strip()]
    query_emb = get_embedding(query)

    similarities = []
    for p in paragraphs:
        p_emb = get_embedding(p)
        sim = np.dot(query_emb, p_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(p_emb))
        similarities.append((sim, p))

    # Ordenar por similitud y devolver los más relevantes
    top_contexts = sorted(similarities, key=lambda x: x[0], reverse=True)[:top_k]
    return "\n".join([ctx[1] for ctx in top_contexts])

# 🔹 Generar respuesta final
def get_chat_response(data_in: InputMessage):
    data = data_in.model_dump()
    message = data["message"]

    try:
        knowledge_text = load_knowledge()
        context = retrieve_context(message, knowledge_text)

        prompt = f"""
        Usa la siguiente información como contexto para responder la pregunta del usuario.
        Si la respuesta no está en el texto, responde con "No tengo información suficiente sobre eso."

        Contexto:
        {context}

        Pregunta del usuario:
        {message}
        """

        completion = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un asistente experto que responde siempre en español de forma clara y precisa."},
                {"role": "user", "content": prompt}
            ]
        )

        return completion.choices[0].message.content

    except Exception as e:
        print(f"Error en RAG Service: {e}")
        return "Error al generar respuesta con RAG."
