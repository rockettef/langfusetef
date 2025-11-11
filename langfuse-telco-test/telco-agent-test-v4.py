"""
Telco Customer Service Agent V4 - Enhanced Langfuse Integration
===============================================================================
Major Updates in V4:
- Fixed evaluation scoring with proper trace_id retrieval
- Enhanced error handling and logging
- Improved score creation using SDK v3 methods
- Better context management for evaluations
- Added performance metrics tracking
- Improved code documentation

Author: Madrid Telco AI Team
Date: November 3, 2025
Langfuse Version: v3+
"""

import os
import time
from datetime import datetime
from dotenv import load_dotenv
from langfuse import observe, get_client
import openai

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Azure OpenAI setup
client = openai.AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)

deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

# Langfuse client for advanced features
langfuse = get_client()

# Configuration flags - Enable/disable features for testing
ENABLE_DEBUG_LOGS = True  # Print detailed API responses
ENABLE_LLM_JUDGE = True  # Evaluate responses with GPT-4
ENABLE_SESSION_TRACKING = True  # Group queries by customer session
ENABLE_PROMPT_MANAGEMENT = False  # Use centrally managed prompts (requires setup)

# =============================================================================
# CORE AGENT FUNCTIONS
# =============================================================================

@observe()
def classify_customer_intent(query: str) -> str:
    """
    Classifies customer query into one of four categories.

    Categories:
    - billing: Payment, charges, invoices
    - technical: Network issues, connectivity problems
    - sales: Product inquiries, upgrades, new services
    - general: Account management, other requests

    Args:
        query: Customer's natural language question

    Returns:
        Intent category as lowercase string

    Langfuse: Traces token usage, latency, and model parameters
    """
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
            max_completion_tokens=2000  # Increased for reasoning models
        )

        # Debug logging
        if ENABLE_DEBUG_LOGS:
            print(f"[DEBUG] Classification response: {response}")

        # Safe extraction with error handling
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            if content:
                return content.strip().lower()
            else:
                print("[WARNING] Empty content in classification response")
                print(f"[INFO] Finish reason: {response.choices[0].finish_reason}")
                if hasattr(response.usage, 'completion_tokens_details'):
                    print(f"[INFO] Reasoning tokens: {response.usage.completion_tokens_details.reasoning_tokens}")
                return "general"  # Default fallback
        else:
            print("[ERROR] No choices in classification response")
            return "general"

    except Exception as e:
        print(f"[ERROR] Classification failed: {e}")
        return "general"


@observe()
def generate_response(query: str, intent: str) -> str:
    """
    Generates context-aware customer service response.

    Uses intent-specific system prompts to provide tailored responses:
    - Billing: Empathetic, references Spanish payment methods
    - Technical: Step-by-step troubleshooting
    - Sales: Highlights 5G, fiber, bundles
    - General: Routes to appropriate specialists

    Args:
        query: Customer's question
        intent: Detected intent category

    Returns:
        Generated response string

    Langfuse: Traces full conversation context and response quality
    """
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

    # Start a Langfuse "generation" block to explicitly track tokens/cost for this LLM response
    with langfuse.start_as_current_generation(
        name="openai-style-generation",
        model=deployment  # or use a model name string
    ) as generation:
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": system_prompts.get(intent, system_prompts["general"])},
                    {"role": "user", "content": query}
                ],
                max_completion_tokens=4000
            )

            if ENABLE_DEBUG_LOGS:
                print(f"[DEBUG] Generation response: {response}")

            # Extract actual usage from API (Azure/OpenAI uses 'usage' field)
            usage = getattr(response, "usage", None)
            if usage:
                # Map usage details to OpenAI schema for Langfuse dashboard
                generation.update(usage_details={
                    "prompt_tokens": usage.prompt_tokens or 0,
                    "completion_tokens": usage.completion_tokens or 0,
                    "total_tokens": usage.total_tokens or 0,
                })

            # Safe content extraction
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    return content
                else:
                    print("[WARNING] Empty content in generation response")
                    return "I apologize, but I'm unable to provide a response at this time. Please contact our support team."
            else:
                print("[ERROR] No choices in generation response")
                return "I apologize, but I'm unable to provide a response at this time. Please contact our support team."

        except Exception as e:
            print(f"[ERROR] Response generation failed: {e}")
            return f"Error generating response: {str(e)}"


# =============================================================================
# EVALUATION FUNCTIONS (Phase 1 Scaling) - FIXED IN V4
# =============================================================================

@observe()
def evaluate_response_quality(query: str, response: str, intent: str) -> float:
    """
    Uses LLM-as-judge to evaluate response quality on multiple criteria.

    V4 IMPROVEMENT: Properly retrieves trace_id and creates scores using SDK v3 methods

    Evaluation Criteria:
    - Relevance to customer query (1-5)
    - Accuracy of information (1-5)
    - Professional and empathetic tone (1-5)
    - Completeness of answer (1-5)

    Args:
        query: Original customer question
        response: Agent's generated response
        intent: Detected intent category

    Returns:
        Quality score (0.0 - 1.0)

    Langfuse: Scores are attached to current trace for analysis
    """
    if not ENABLE_LLM_JUDGE:
        return 1.0  # Skip evaluation if disabled

    try:
        judge_prompt = f"""
You are an expert evaluator for telecom customer service responses.

Customer Query: {query}
Detected Intent: {intent}
Agent Response: {response}

Evaluate the response on a scale of 1-5 for each criterion:
1. Relevance to query
2. Accuracy of information
3. Professional and empathetic tone
4. Completeness

Provide only the average score as a single number (e.g., 4.2).
"""

        eval_response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": judge_prompt}],
            max_completion_tokens=500
        )

        if eval_response.choices and eval_response.choices[0].message.content:
            score_text = eval_response.choices[0].message.content.strip()
            try:
                raw_score = float(score_text)
                normalized_score = raw_score / 5.0  # Normalize to 0-1

                # V4 FIX: Use score_current_trace() to properly attach score to the current trace
                # This method works within the @observe() context and automatically gets the trace_id
                langfuse.score_current_trace(
                    name="llm_judge_quality",
                    value=normalized_score,
                    data_type="NUMERIC",
                    comment=f"Quality evaluation for {intent} query (raw score: {raw_score:.1f}/5.0)"
                )

                print(f"[EVAL] ✅ Quality score: {normalized_score:.2f} ({raw_score:.1f}/5.0)")
                print(f"[EVAL] Score successfully attached to trace")
                return normalized_score
            except ValueError as ve:
                print(f"[EVAL] ⚠️ Could not parse score: {score_text}, error: {ve}")
                return 0.5
        else:
            print("[EVAL] ⚠️ Failed to get evaluation score - empty response")
            return 0.5

    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.5


# =============================================================================
# MAIN AGENT WORKFLOW - ENHANCED IN V4
# =============================================================================

@observe()
def telco_agent_workflow(customer_query: str, customer_id: str = None, session_id: str = None) -> dict:
    """
    Main orchestration function for telco customer service agent.

    V4 IMPROVEMENTS:
    - Better error handling throughout workflow
    - Enhanced logging with emojis for readability
    - Proper trace and score management
    - Performance metrics tracking

    Workflow:
    1. Classify customer intent
    2. Generate context-aware response
    3. Evaluate response quality (if enabled)
    4. Track session and user metadata
    5. Log all steps to Langfuse for observability

    Args:
        customer_query: Customer's natural language question
        customer_id: Optional customer identifier for personalization
        session_id: Optional session ID for multi-turn tracking

    Returns:
        Dictionary with query, intent, response, and metadata

    Langfuse: Creates hierarchical trace with all sub-operations
    """
    start_time = time.time()

    # Update trace metadata for session tracking
    if ENABLE_SESSION_TRACKING and (customer_id or session_id):
        langfuse.update_current_trace(
            user_id=customer_id or "anonymous",
            session_id=session_id or "test-session",
            tags=["b2c", "spain", "customer-service", "v4"],
            metadata={
                "timestamp": datetime.now().isoformat(),
                "channel": "test",
                "language": "es",
                "version": "v4"
            }
        )

    print(f"\n{'='*60}")
    print(f"🔵 Customer Query: {customer_query}")
    if customer_id:
        print(f"👤 Customer ID: {customer_id}")
    if session_id:
        print(f"🔗 Session ID: {session_id}")
    print(f"{'='*60}")

    # Step 1: Classify intent
    intent = classify_customer_intent(customer_query)
    print(f"→ 🎯 Detected Intent: {intent}")

    # Step 2: Generate response
    response = generate_response(customer_query, intent)
    print(f"\n→ 💬 Agent Response:\n{response}")

    # Step 3: Calculate metrics
    end_time = time.time()
    latency = end_time - start_time
    print(f"\n⏱️  Total Latency: {latency:.2f}s")

    # Step 4: Evaluate quality (if enabled)
    # V4 FIX: Call evaluate_response_quality which now properly scores the trace
    quality_score = None
    if ENABLE_LLM_JUDGE and response and "unable to provide" not in response.lower():
        print(f"\n🔍 Running quality evaluation...")
        quality_score = evaluate_response_quality(customer_query, response, intent)

    print(f"{'='*60}\n")

    # Return structured result
    return {
        "query": customer_query,
        "intent": intent,
        "response": response,
        "latency_seconds": latency,
        "quality_score": quality_score,
        "timestamp": datetime.now().isoformat()
    }


# =============================================================================
# DATASET MANAGEMENT (Phase 2 Scaling)
# =============================================================================

def create_evaluation_dataset(dataset_name: str = "telco-customer-queries-v1"):
    """
    Creates a Langfuse dataset with representative test cases.

    Use this for:
    - Regression testing before deployments
    - A/B testing different prompt versions
    - Benchmarking model improvements

    Args:
        dataset_name: Unique identifier for the dataset

    Langfuse: Dataset persists in UI for team collaboration
    """
    print(f"\n📊 Creating dataset: {dataset_name}")

    try:
        # Create dataset
        langfuse.create_dataset(
            name=dataset_name,
            description="Representative customer service queries for Telefónica Madrid"
        )

        # Define test cases with expected outputs
        test_cases = [
            {
                "input": {"query": "Why is my bill 20 euros higher this month?"},
                "expected_output": {"intent": "billing"},
                "metadata": {"difficulty": "medium", "priority": "high"}
            },
            {
                "input": {"query": "My internet keeps disconnecting every few minutes"},
                "expected_output": {"intent": "technical"},
                "metadata": {"difficulty": "high", "priority": "critical"}
            },
            {
                "input": {"query": "Do you have 5G coverage in Malasaña neighborhood?"},
                "expected_output": {"intent": "sales"},
                "metadata": {"difficulty": "low", "priority": "medium"}
            },
            {
                "input": {"query": "How do I change my password?"},
                "expected_output": {"intent": "general"},
                "metadata": {"difficulty": "low", "priority": "low"}
            },
            {
                "input": {"query": "I want to cancel my contract"},
                "expected_output": {"intent": "billing"},
                "metadata": {"difficulty": "high", "priority": "critical"}
            }
        ]

        # Add items to dataset
        for case in test_cases:
            langfuse.create_dataset_item(
                dataset_name=dataset_name,
                input=case["input"],
                expected_output=case["expected_output"],
                metadata=case["metadata"]
            )

        print(f"✅ Created dataset with {len(test_cases)} test cases")
        print(f"📊 View in Langfuse: http://localhost:3000/datasets/{dataset_name}")

    except Exception as e:
        print(f"[ERROR] Failed to create dataset: {e}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 TELCO CUSTOMER SERVICE AGENT V4 - ENHANCED LANGFUSE INTEGRATION")
    print("="*80)
    print(f"📅 Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Model: {deployment}")
    print(f"🔍 Debug logs: {'✅ Enabled' if ENABLE_DEBUG_LOGS else '❌ Disabled'}")
    print(f"⚖️  LLM Judge: {'✅ Enabled' if ENABLE_LLM_JUDGE else '❌ Disabled'}")
    print(f"👥 Session tracking: {'✅ Enabled' if ENABLE_SESSION_TRACKING else '❌ Disabled'}")
    print("="*80 + "\n")

    print("📝 V4 IMPROVEMENTS:")
    print("  ✅ Fixed evaluation scoring with proper trace_id retrieval")
    print("  ✅ Enhanced error handling and logging")
    print("  ✅ Improved score creation using SDK v3 methods")
    print("  ✅ Better context management for evaluations")
    print("  ✅ Added performance metrics tracking")
    print("="*80 + "\n")

    # Test queries representing diverse customer service scenarios
    test_queries = [
        "Do you have 5G coverage in Malasaña neighborhood?",
       # "How do I change my password?",
        #"Why is my bill 20 euros higher this month?"
    ]

    # Simulate a customer session
    session_id = f"session-{int(time.time())}"
    customer_id = "customer-12345"
    results = []

    # Process each query in the session
    for idx, query in enumerate(test_queries, 1):
        print(f"\n[{idx}/{len(test_queries)}] Processing query...")
        result = telco_agent_workflow(
            customer_query=query,
            customer_id=customer_id,
            session_id=session_id
        )
        results.append(result)

        # Small delay between queries for readability
        if idx < len(test_queries):
            time.sleep(1)

    # =============================================================================
    # SESSION SUMMARY
    # =============================================================================

    print("\n" + "="*80)
    print("📊 SESSION SUMMARY")
    print("="*80)

    # Calculate aggregate metrics
    total_latency = sum(r['latency_seconds'] for r in results)
    avg_latency = total_latency / len(results)

    if ENABLE_LLM_JUDGE:
        quality_scores = [r['quality_score'] for r in results if r['quality_score']]
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            print(f"📈 Average Quality Score: {avg_quality:.2f} ({avg_quality*5:.1f}/5.0)")
            print(f"📈 Quality Scores Range: {min(quality_scores):.2f} - {max(quality_scores):.2f}")

    print(f"⏱️  Total Latency: {total_latency:.2f}s")
    print(f"⏱️  Average Latency: {avg_latency:.2f}s per query")
    print(f"🎯 Queries Processed: {len(results)}")

    # Intent distribution
    intents = {}
    for r in results:
        intent = r['intent']
        intents[intent] = intents.get(intent, 0) + 1

    print(f"\n📊 Intent Distribution:")
    for intent, count in intents.items():
        print(f"   - {intent}: {count} ({count/len(results)*100:.0f}%)")

    print("\n" + "="*80)
    print("✅ All test queries processed successfully!")
    print(f"📊 View detailed traces: http://localhost:3000/traces")
    print(f"👥 View session: http://localhost:3000/sessions/{session_id}")
    print(f"💰 View cost analysis: http://localhost:3000/dashboard")
    print(f"📈 View scores: http://localhost:3000/scores")
    print("="*80 + "\n")

    # Flush Langfuse events to ensure all data is sent
    print("🔄 Flushing Langfuse events...")
    langfuse.flush()
    print("✅ Langfuse events successfully flushed!\n")

    # =============================================================================
    # NEXT STEPS
    # =============================================================================

    print("\n" + "="*80)
    print("🚀 NEXT STEPS FOR SCALING")
    print("="*80)
    print("\n1️⃣  Create Evaluation Dataset:")
    print("   Run: create_evaluation_dataset()")
    print("   Use for regression testing before deployments\n")

    print("2️⃣  Analyze Scores in Langfuse:")
    print("   - Check the Scores tab in Langfuse UI")
    print("   - Filter by score name: 'llm_judge_quality'")
    print("   - View score distribution and trends\n")

    print("3️⃣  Enable Production Features:")
    print("   - Integrate with real customer database")
    print("   - Add alerting for quality/cost thresholds")
    print("   - Set up continuous monitoring\n")

    print("4️⃣  Scale Infrastructure:")
    print("   - Deploy to production environment")
    print("   - Configure auto-scaling")
    print("   - Set up EU data residency for GDPR\n")
    print("="*80)

    # Uncomment to create dataset
    # create_evaluation_dataset()

    
    # Uncomment to see prompt management setup
    # setup_prompt_management()