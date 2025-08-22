#!/usr/bin/env python3
"""Debug script to check what the ollama library returns."""

import asyncio
from ollama import AsyncClient
import json

async def debug_ollama_library():
    """Check what the ollama library actually returns."""
    print("Testing ollama library response...")
    
    client = AsyncClient(host="http://localhost:11434")
    
    try:
        # Call list() method
        response = await client.list()
        
        print(f"\nResponse type: {type(response)}")
        
        if hasattr(response, '__dict__'):
            print(f"Response attributes: {response.__dict__}")
        
        # Try different ways to access the data
        if isinstance(response, dict):
            print("\nResponse is a dict:")
            print(json.dumps(response, indent=2, default=str))
        else:
            print("\nResponse is not a dict")
            print(f"Response: {response}")
            
            # Check if it has models attribute
            if hasattr(response, 'models'):
                print(f"\nHas 'models' attribute: {response.models}")
                if response.models:
                    first_model = response.models[0]
                    print(f"\nFirst model type: {type(first_model)}")
                    if hasattr(first_model, '__dict__'):
                        print(f"First model attributes: {first_model.__dict__}")
            
            # Try to iterate
            try:
                print("\nTrying to iterate response:")
                for item in response:
                    print(f"  Item type: {type(item)}")
                    if hasattr(item, '__dict__'):
                        print(f"  Item attributes: {item.__dict__}")
                    else:
                        print(f"  Item: {item}")
                    break  # Just check first item
            except TypeError:
                print("  Response is not iterable")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_ollama_library())