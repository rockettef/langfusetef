import os
from dotenv import load_dotenv
from langfuse import observe, get_client
import openai

load_dotenv()

# Azure OpenAI setup
client = openai.AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

@observe()
def classify_customer_intent(query: str) -> str:
    """Classifies customer query into billing, technical, sales, or general"""
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": "You are a telecom customer service classifier. "
                              "Classify the query into: billing, technical, sales, or general. "
                              "Respond with only the category name."
                },
                {"role": "user", "content": query}
            ],
            max_completion_tokens=1000
        )
        
        # Debug: print full response
        print(f"[DEBUG] Classification response: {response}")
        
        # Safe extraction with error handling
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content:
                return content.strip().lower()
            else:
                print("[WARNING] Empty content in classification response")
                return "general"  # Default fallback
        else:
            print("[ERROR] No choices in classification response")
            return "general"
            
    except Exception as e:
        print(f"[ERROR] Classification failed: {e}")
        return "general"

@observe()
def generate_response(query: str, intent: str) -> str:
    """Generates customer service response based on detected intent"""
    system_prompts = {
        "billing": "You are a helpful billing specialist for a Madrid-based telecom. "
                  "Provide clear, concise answers about bills, charges, and payments. "
                  "Be empathetic and reference Spanish payment methods.",
        "technical": "You are a technical support specialist. "
                    "Provide step-by-step troubleshooting for mobile and internet issues. "
                    "Be clear and patient.",
        "sales": "You are a sales representative. "
                "Explain our 5G plans, fiber internet, and bundled services. "
                "Mention competitive pricing.",
        "general": "You are a helpful customer service representative. "
                  "Provide general assistance and route to specialists when needed."
    }
    
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompts.get(intent, system_prompts["general"])},
                {"role": "user", "content": query}
            ],
            max_completion_tokens=3000
        )
        
        # Debug: print full response
        print(f"[DEBUG] Generation response: {response}")
        
        # Safe extraction with error handling
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content:
                return content
            else:
                print("[WARNING] Empty content in generation response")
                # Check if content was filtered
                if hasattr(response.choices[0], 'content_filter_results'):
                    print(f"[FILTER] Content filter results: {response.choices[0].content_filter_results}")
                return "I apologize, but I'm unable to provide a response at this time. Please contact our support team."
        else:
            print("[ERROR] No choices in generation response")
            return "I apologize, but I'm unable to provide a response at this time. Please contact our support team."
            
    except Exception as e:
        print(f"[ERROR] Response generation failed: {e}")
        return f"Error generating response: {str(e)}"

@observe()
def telco_agent_workflow(customer_query: str):
    """Main agent workflow with full observability"""
    print(f"\n{'='*60}")
    print(f"Customer Query: {customer_query}")
    print(f"{'='*60}")
    
    intent = classify_customer_intent(customer_query)
    print(f"→ Detected Intent: {intent}")
    
    response = generate_response(customer_query, intent)
    print(f"\n→ Agent Response:\n{response}")
    print(f"{'='*60}\n")
    
    return {
        "query": customer_query,
        "intent": intent,
        "response": response
    }

if __name__ == "__main__":
    test_queries = [
        "Why is my bill 20 euros higher this month?",
        "My internet keeps disconnecting every few minutes",
        "Do you have 5G coverage in Malasaña neighborhood?",
        "How do I change my password?"
    ]
    
    print("\n🚀 Starting Telco Agent Evaluation with Langfuse Tracing\n")
    
    for query in test_queries:
        telco_agent_workflow(query)
    
    print("✅ All test queries processed!")
    print("📊 View traces at: http://localhost:3000")
    
    # Flush Langfuse events
    langfuse = get_client()
    langfuse.flush()
    print("✅ Langfuse events flushed!")