import streamlit as st

class SideBar:
    def __init__(self):
        self.urls = []
        self.title = "News Articles URL's"
        self.process_url_clicked = []
    
    def create(self):
        st.sidebar.title(self.title)
        for i in range(3):
            url = st.sidebar.text_input(f"URL {i + 1}")
            self.urls.append(url)
        self.process_url_clicked = st.sidebar.button("Process URLs")

class mainElement():
    def __init__(self):
        self.placeholder = st.empty()
        self.title = []
        self.query = []

    def create(self):
        self.title = st.title("News Research tool 📈")
        self.placeholder = st.empty()

    def askQuestion(self):
        self.query = self.placeholder.text_input("Question: ")

    def printAnswer(self, result):
        st.header("Answer")
        st.write(result["answer"])
    
    def printSources(self, result):
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)