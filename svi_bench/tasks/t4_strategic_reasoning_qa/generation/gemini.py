from google import genai
import time
GEMINI_API_KEY=None

def gemini(model, prompt):
    client = genai.Client(api_key=GEMINI_API_KEY)
    for i in range(30):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    automatic_function_calling=genai.types.AutomaticFunctionCallingConfig(disable=True)
                )
            )
            return response.text
        except Exception as e:
            if i % 5 == 0:
                print(f"attempt {i} failed: {e}")
            time.sleep(120)
    
    raise Exception("failed after 30 tries")
