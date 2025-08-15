# llm_advisor/advisor.py
import re

class BusinessAdvisor:
    def __init__(self):
        pass

    def get_advice(self, context):
        advice = "📊 Business Insight:\n"
        margin_match = re.search(r'Profit margin: ([\d.]+)', context)

        if margin_match:
            margin = float(margin_match.group(1))
            if margin < 10:
                advice += "- Low margin: Negotiate vendor costs or raise price.\n"
            elif margin < 20:
                advice += "- Moderate margin: Look for efficiency improvements.\n"
            else:
                advice += "- Strong margin: Consider investing in growth.\n"

        if "competitor price" in context.lower():
            advice += "- Differentiate on service, not just price.\n"

        advice += "- Monitor competitor pricing weekly.\n"
        advice += "- Build loyalty programs to reduce churn."

        return advice

    def chat(self, question, context):
        question = question.lower()

        # Check if user is asking about margin
        if "margin" in question:
            margin_match = re.search(r'Profit margin: ([\d.]+)', context)
            if margin_match:
                margin = float(margin_match.group(1))
                if margin < 10:
                    return "Your profit margin is low. Try renegotiating vendor prices or increasing your selling price."
                elif margin < 20:
                    return "Your margin is moderate. Consider operational efficiencies to improve profitability."
                else:
                    return "Your margin is strong — you could invest in growth or marketing."
            else:
                return "I couldn't find margin information in the context."

        # Check if asking about competitors
        elif "competitor" in question:
            return "Monitor competitor prices regularly and add value beyond price, like better service or faster delivery."

        # Check if asking about pricing strategy
        elif "price" in question:
            return "Aim to price competitively but maintain your minimum margin for profitability."

        # Generic fallback
        else:
            return "I can help with pricing, margin analysis, and competitor strategies. Could you give me more details?"
