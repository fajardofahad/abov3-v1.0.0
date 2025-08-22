#!/usr/bin/env python3
"""Debug script to check Ollama API response structure."""

import asyncio
import httpx
import json

async def debug_ollama_api():
    """Check the actual response from Ollama API."""
    print("Checking Ollama API response structure...")
    
    async with httpx.AsyncClient() as client:
        try:
            # Call the Ollama API directly
            response = await client.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            
            data = response.json()
            print("\nRaw API Response:")
            print(json.dumps(data, indent=2))
            
            # Check structure
            if "models" in data:
                models = data["models"]
                print(f"\nFound {len(models)} models")
                
                if models:
                    print("\nFirst model structure:")
                    first_model = models[0]
                    print(f"Type: {type(first_model)}")
                    print(f"Keys: {list(first_model.keys())}")
                    print(f"Content: {json.dumps(first_model, indent=2)}")
            else:
                print("\nNo 'models' key in response")
                print(f"Available keys: {list(data.keys())}")
                
        except httpx.HTTPStatusError as e:
            print(f"HTTP Error: {e}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_ollama_api())