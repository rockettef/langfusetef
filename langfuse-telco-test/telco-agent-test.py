import os
from dotenv import load_dotenv
from langfuse import observe
import openai

load_dotenv()

# Azure OpenAI setup (openai>=1.0.0 style)
client = openai.AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

@observe()
def classify_customer_intent(query: str) -> str:
    response = client.chat.completions.create(
        model=deployment,  # Azure: 'model' is the deployment name (not the model family)
        messages=[
            {
                "role": "system",
                "content": "You are a telecom customer service classifier. Classify the query into: billing, technical, sales, or general. Respond with only the category name."
            },
            {"role": "user", "content": query}
        ],
       # temperature=0.1,
        # Azure OpenAI deployments expect `max_completion_tokens` instead of `max_tokens`
        max_completion_tokens=500
    )
    # print(response)
    return response.choices[0].message.content.strip().lower()

@observe()
def generate_response(query: str, intent: str) -> str:
    system_prompts = {
        "billing": "You are a helpful billing specialist for a Madrid-based telecom. Provide clear, concise answers about bills, charges, and payments. Be empathetic and reference Spanish payment methods.",
        "technical": "You are a technical support specialist. Provide step-by-step troubleshooting for mobile and internet issues. Be clear and patient.",
        "sales": "You are a sales representative. Explain our 5G plans, fiber internet, and bundled services. Mention competitive pricing.",
        "general": "You are a helpful customer service representative. Provide general assistance and route to specialists when needed."
    }
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompts.get(intent, system_prompts["general"])},
            {"role": "user", "content": query}
        ],
     #   temperature=0.7,
        # Azure OpenAI deployments expect `max_completion_tokens` instead of `max_tokens`
        max_completion_tokens=500
    )
    # print(response)
    return response.choices[0].message.content

@observe()
def telco_agent_workflow(customer_query: str):
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
