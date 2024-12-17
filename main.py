import os
import streamlit as st
import pickle
import time
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)
# App Title
st.title("FinBot: Equity Insight Navigator 📈")
st.sidebar.title("News Article URLs")

# Collect URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

# Define company and ticker mapping
companies = ["Tata Motors", "Tesla", "Ford", "General Motors", "Toyota"]
ticker_symbol_map = {
    "Tesla": "TSLA",
    "Tata Motors": "TATAMOTORS.NS",
    "Ford": "F",
    "Toyota": "TM",
    "General Motors": "GM"
}

# Initialize LLM
llm = OpenAI(temperature=0.9, max_tokens=500)

# Function to tag companies in documents
def tag_company(document, companies):
    for company in companies:
        if company.lower() in document.page_content.lower():
            document.metadata['company'] = company
            return document
    document.metadata['company'] = "Unknown"
    return document

# Function to fetch stock data and generate line charts
def plot_stock_data(company_name, ticker_symbol):
    st.subheader(f"Stock Price Trends for {company_name}")
    
    # Allow the user to select the chart style
    chart_type = st.selectbox("Select Chart Style", ["Standard Line Chart", "Step Line Chart"], key=company_name)
    
    if ticker_symbol:
        try:
            # Fetch stock data
            stock_data = yf.download(ticker_symbol, period="6mo", interval="1wk")
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Extract the data
            dates = stock_data.index
            closing_prices = stock_data['Close']

            # Plot chart based on user selection
            if chart_type == "Standard Line Chart":
                plt.plot(dates, closing_prices, marker='o', linestyle='-', color='tab:blue', label='Closing Price')
            elif chart_type == "Step Line Chart":
                plt.step(dates, closing_prices, where='mid', color='tab:orange', label='Step Closing Price')

            # Add chart titles and labels
            plt.title(f"{company_name} - Stock Price Trends")
            plt.xlabel("Date")
            plt.ylabel("Price (in INR)")
            plt.xticks(rotation=45)
            plt.legend()

            # Render the chart in Streamlit
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error fetching stock data: {e}")
    else:
        st.warning(f"No ticker symbol found for {company_name}. Displaying mock data.")
        
        # Generate mock data for unknown companies
        mock_data = pd.DataFrame({
            "Date": pd.date_range(start="2023-01-01", periods=12, freq="M"),
            "Price": [100 + i * 10 + (-10)**(i % 2) * 5 for i in range(12)]
        })
        
        fig, ax = plt.subplots(figsize=(10, 5))
        dates = mock_data["Date"]
        prices = mock_data["Price"]

        # Plot mock data based on selected style
        if chart_type == "Standard Line Chart":
            plt.plot(dates, prices, marker='o', linestyle='-', color='tab:green', label='Mock Data')
        elif chart_type == "Step Line Chart":
            plt.step(dates, prices, where='mid', color='tab:red', label='Step Mock Data')

        # Add chart titles and labels
        plt.title(f"{company_name} - Mock Stock Prices")
        plt.xlabel("Month")
        plt.ylabel("Price (in INR)")
        plt.xticks(rotation=45)
        plt.legend()

        # Render the chart in Streamlit
        st.pyplot(fig)


# Process URLs
if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    st.info("Loading data... Please wait.")
    data = loader.load()

    # Tag documents with company names
    tagged_data = [tag_company(doc, companies) for doc in data]

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
    docs = text_splitter.split_documents(tagged_data)
    
    # Create embeddings and FAISS vector store
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    
    # Save vector store
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)
    st.success("Data processed and embeddings created successfully!")

# User Query Section
query = st.text_input("Ask a Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            
            # Detect company name from query
            def detect_company(query, companies):
                for company in companies:
                    if company.lower() in query.lower():
                        return company
                return "Unknown"
            
            company = detect_company(query, companies)

            # Retrieve and filter documents
            docs = retriever.get_relevant_documents(query)
            filtered_docs = [doc for doc in docs if doc.metadata.get('company', '') == company]
            
            if not filtered_docs:
                st.warning("No specific data found for this company. Displaying general results.")
                filtered_docs = docs  # Fallback

            # Create QA chain
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            
            # Display Answer
            st.header("Answer")
            st.write(result["answer"])
            
            # Display related insights
            st.subheader("Related Financial Insights:")
            st.write(f"Showing data for: **{company}**")

            # Display the stock price trends
            ticker_symbol = ticker_symbol_map.get(company)
            plot_stock_data(company, ticker_symbol)
    else:
        st.error("No FAISS database found. Please process URLs first.")

