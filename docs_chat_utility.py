import os
import fitz  # PyMuPDF
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain

working_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize the LLM (Note: Make sure the Ollama library supports integration with Langchain)
llm = Ollama(
    model="llama3.2",
    temperature=0.2,
)

embeddings = HuggingFaceEmbeddings()

def get_answer(file_path, query):
    try:
        # Step 1: Load PDF and extract text
        doc_text = ""
        with fitz.open(file_path) as pdf:
            for page_num in range(pdf.page_count):
                page = pdf[page_num]
                doc_text += page.get_text("text")  # Extract text from each page
        
        print(f"Total Extracted Text Length: {len(doc_text)} characters")

        # Step 2: Check if text was extracted
        if not doc_text.strip():
            return "No text found in the uploaded PDF file."

        # Step 3: Define chunking parameters
        chunk_size = 1000 if len(doc_text) > 1000 else len(doc_text)
        chunk_overlap = 200 if chunk_size > 200 else max(0, chunk_size - 1)
        print(f"Chunk Size: {chunk_size}, Chunk Overlap: {chunk_overlap}")

        # Step 4: Split the document into chunks
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        text_chunks = text_splitter.split_text(doc_text)

        # Print number of chunks for verification
        print(f"Number of Text Chunks: {len(text_chunks)}")
        
        if not text_chunks:
            return "Failed to split the document into chunks. The document may be too small."

        # Step 5: Create a FAISS knowledge base from the text chunks
        knowledge_base = FAISS.from_texts(text_chunks, embeddings)

        # Step 6: Define prompt for combining document chunks
        combine_prompt_template = """Use the information below to provide a concise response to the user's question.
        Document Context: {context}
        User Question: {question}
        """
        combine_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=combine_prompt_template
        )

        # Step 7: Create the combine_docs_chain with the prompt
        combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=combine_prompt)

        # Step 8: Set up the question generation chain (use LLMChain as a question generator)
        question_prompt_template = """Based on the following context, generate a question for the assistant.
        Context: {context}
        """
        question_prompt = PromptTemplate(input_variables=["context"], template=question_prompt_template)
        question_generator = LLMChain(llm=llm, prompt=question_prompt)

        # Step 9: Set up the ConversationalRetrievalChain with retriever, question generator, and combine_docs_chain
        qa_chain = ConversationalRetrievalChain(
            retriever=knowledge_base.as_retriever(),
            question_generator=question_generator,
            combine_docs_chain=combine_docs_chain
        )

        # Step 10: Query the QA chain and handle the response correctly
        response = qa_chain.run(query)

        # Print response for debugging
        print(f"Response: {response}")
        
        # Return the answer directly since `.run` provides a usable format
        return response if response else "No answer found for the query."
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return f"An error occurred: {str(e)}"

# Example usage (you can test with your file and query)
# file_path = "path/to/your/file.pdf"
# query = "What is the main point of the paper?"
# print(get_answer(file_path, query))
