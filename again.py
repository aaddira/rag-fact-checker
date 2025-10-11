import json
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import os
import time

EMBEDDING_MODEL_NAME="sentence-transformers/all-mpnet-base-v2"
READER_MODEL_NAME="mistralai/Mixtral-8x7B-v0.1"
RERANKER_MODEL_NAME="cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDINGS_FILE="embeddings.pkl"
SYS_PROMPT = (
        "You are an expert Q&A assistant. Use the following pieces of retrieved "
        "context to answer the user's question concisely. "
        "If the answer cannot be found in the context, state that you do not "
        "have enough information, but do not make up an answer."
    )
INITIAL_K=20
FINAL_K=5

class RAGPipeline:
    def __init__(self, sys_prompt, documents, embedding_model, reader_model, reranker_model, initial_k, final_k):
        self.initial_k=initial_k
        self.final_k=final_k
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.embedder=SentenceTransformer(embedding_model, device=self.device)
        self.reranker=CrossEncoder(reranker_model, device=self.device)
        self.reader_model=reader_model
        self.contents=[doc['Content'] for doc in documents]
        self.sys_prompt=sys_prompt

        if os.path.exists(EMBEDDINGS_FILE):
            self.embeddings=self._load_embeddings()
        else:
            self.embeddings=self._embed()

        self.tokenizer=AutoTokenizer.from_pretrained(reader_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token 

        self.reader=AutoModelForCausalLM.from_pretrained(
            reader_model,
            torch_dtype=torch.bfloat16 if self.device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16,
            device_map='auto'
        )

        self.generator=pipeline(
            "text-generation",
            model=self.reader,
            tokenizer=self.tokenizer,
            torch_dtype=self.reader.dtype,
            device_map="auto"
        )

    def _embed(self):
        self.embeddings=self.embedder.encode(self.contents, convert_to_tensor=True, device=self.device)
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(self.embeddings.cpu(), f)
    
    def _load_embeddings(self):
        with open(EMBEDDING_MODEL_NAME, 'rb') as f:
            embeddings_cpu = pickle.load(f)
        return embeddings_cpu.to(self.device) 
    
    def _retrieve(self, query_vector, original_query):
        query_vector_np=query_vector.cpu().numpy().reshape(1,-1)
        embeddings_np=self.embeddings.cpu().numpy()
        sims=cosine_similarity(query_vector_np,embeddings_np)[0]
        initial_indices=np.argsort(sims)[::-1][self.initial_k]
        initial_info=[self.documents[i] for i in initial_indices]
        #reranking time
        pairs=[(original_query, doc['Content']) for doc in initial_info]
        reranked=self.reranker.predict(pairs)
        final_top=np.argsort(reranked)[::-1][:self.final_k]
        final_info=[initial_info[i] for i in final_top]
        return final_info

    def _prompt_format(self, query, final_info):
        context="\n\n".join([f"Document {i+1}: {doc['Content']}" for i, doc in enumerate(final_info)])
        complete_query=(
            f"{self.sys_prompt}\n\n"
            f"--- CONTEXT ---\n{context}\n\n"
            f"--- USER QUESTION ---\n{query}"
        )
        messages=[
            {"role": "user", "content": complete_query}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt
    
    def query(self, query, max_tokens=500):
        print(f"Query: {query}")
        start_time = time.time()
        embedded_query=self.embedder.encode(query, convert_to_tensor=True, device=self.device)
        retrieved_info=self._retrieve(embedded_query,query)
        retrieval_time = time.time()
        print(f"Retrieved {len(retrieved_info)} documents in {retrieval_time - start_time:.2f}s:")
        for i, doc in enumerate(retrieved_info):
            print(f"  {i+1}. ID {doc['ID']} | Title: {doc['Title']} | Snippet: \"{doc['Content'][:50]}...\"")
        prompt=self._prompt_format(query,retrieved_info)
        generation_output = self.generator(
            prompt,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )
        full_response = generation_output[0]['generated_text']
        try:
            if full_response.startswith(prompt):
                answer = full_response[len(prompt):].strip()
            else:
                answer = full_response.split('[/INST]')[-1].strip()
        except Exception as e:
             answer = f"Error extracting answer: {e}"
        generation_time = time.time()
        print(f"Generation completed in {generation_time - retrieval_time:.2f}s.")
        return answer

if __name__ == "__main__":
    with open("chunks.json", 'rb') as f:
        documents = json.load(f)
    rag = RAGPipeline(sys_prompt=documents, embedding_model=EMBEDDING_MODEL_NAME, reranker_model=RERANKER_MODEL_NAME, reader_model=READER_MODEL_NAME, initial_k=INITIAL_K, final_k=FINAL_K)
    sample_query = "What happened to Louis Desaix after he was recalled from Upper Egypt?" 
    print(f"\nSample Query: {sample_query}\n")
    answer = rag.query(sample_query)    
    print(f"\nFinal Answer: {answer}")

    sample_query_2 = "Where was Desaix born and who were his parents?"
    print(f"\nSample Query: {sample_query_2}\n")
    answer_2 = rag.query(sample_query_2)
    print(f"\nFinal Answer: {answer_2}")

