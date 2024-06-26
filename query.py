from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
import json

load_dotenv()


def get_summary(message: str):
    """
    This function will take in a user query related to travelling and do the following
    1. Find out what is the ask. Is user asking for a stay information, transport information, activities information.
    2. Find out which location user is talking about.
    3. Find out if user has given the sc for the trip - pax_count, travel_dates, from_location, to_location.
    """

    print("User question: ", message)
    print("Asking LLM for the summary..")

    summary_template = """
    You are a travel agent who helps a user with any travel query they have. 
    Given the user question {question} first you need to extract certain information:
    1. Find out what is user asking about. Is it a query regarding stay, transport options, activities or anything else.
        a. give the query_type as 'stay', 'transport', 'activity'. you can create new query_type but try to stick to
        these ones only
        b. give a sub_query_type. Like for query_type = 'stay' sub_query_types can be - 'hotel', 'hostel', 'homestay', 
        for query_type = 'transport' sub_query_types can be - 'flight', 'train', 'bus'
        
        If you are not able to figure out or you are not sure about sub_query_type, give null as value, don't give any 
        garbage value to it.
        
        In output return a JSON with query_type, sub_query_type and query_summary 
        query_summary should not exceed more than 20 words, keep it as precise as possible
        
    2. Find out which location user is talking about in the query. 
    Location must be a city or state or country.
    Don't consider any other nouns like Hotel Names as a location. Return only if you are sure about it.  
    If you are able to figure out then put the same in the previous json with a key name - location, 
    if not then just put null in the same key    
    
    3. Find out SC information.
    What is SC information? It contains the following - 
        a. pax_count - head count of travellers, how many number or travellers are there
        b. travel_dates - start_date or end_date for the trip if user has provided any information of it. this has to be
         a valid date in DD/MM/YYYY format.
        c. from_location - from where user wants to start the trip
        d. to_location - where user want's to go for trip or for which location user is querying about.
    
    From the given user message try to find out the following four information and put them in a JSON format in a key 
    called 'sc' in the same JSON that has other two details from above
    
    if you are not able to extract any of the information, pass null as value to it
    """

    summary_prompt_template = PromptTemplate(input_variables=["question"], template=summary_template)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = summary_prompt_template | llm

    res = chain.invoke(input={"question": message}).json()

    parsed = json.loads(json.loads(res)["content"])

    print("LLM Response:", json.dumps(parsed, indent=2))


if __name__ == "__main__":
    print("Running the program..")
    # get_summary("Show me hotels around Baga Beach in Goa")
    get_summary("Show me stay options around Baga Beach in Goa")
    # get_summary("What are the attraction points near Resort Coco Cabana")
    # get_summary("What are the attraction points near Resort Coco Cabana in Goa")
    # get_summary("What are the attraction points near Resort Coco Cabana in Goa for family of 4 in mid july")
    # get_summary("Best attractions to be done in Goa in month of july")
    print("Program finished..")
