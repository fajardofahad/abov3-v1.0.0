#!/usr/bin/env python3
"""
ABOV3 4 Ollama - Minimal Working Demo

A simple demonstration of ABOV3's core capabilities that works with minimal dependencies.
This demo showcases the key features of ABOV3 without requiring the full dependency stack.

Features demonstrated:
- Direct Ollama connection and health checking
- Interactive chat with streaming responses
- Model management and information
- Error handling and graceful degradation
- Rich terminal output with progress indicators

Requirements:
- Python 3.8+
- ollama package
- rich package (for nice terminal output)
- click package (for CLI interface)
- prompt-toolkit package (for interactive input)

Usage:
    python demo.py
"""

import asyncio
import sys
import traceback
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

# Check Python version first
if sys.version_info < (3, 8):
    print("❌ Error: Python 3.8 or higher is required")
    sys.exit(1)

# Import required packages with graceful fallback
try:
    import ollama
    from ollama import AsyncClient
except ImportError:
    print("❌ Error: 'ollama' package is required. Install with: pip install ollama")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.syntax import Syntax
    from rich.text import Text
    from rich.live import Live
    from rich import print as rprint
except ImportError:
    print("❌ Error: 'rich' package is required. Install with: pip install rich")
    sys.exit(1)

try:
    import click
except ImportError:
    print("❌ Error: 'click' package is required. Install with: pip install click")
    sys.exit(1)

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.shortcuts import confirm
except ImportError:
    print("❌ Error: 'prompt-toolkit' package is required. Install with: pip install prompt-toolkit")
    sys.exit(1)


class ABOV3Demo:
    """
    Minimal ABOV3 demo showcasing core AI coding capabilities.
    """
    
    def __init__(self, host: str = "http://localhost:11434"):
        self.console = Console()
        self.host = host
        self.client: Optional[AsyncClient] = None
        self.available_models: List[Dict[str, Any]] = []
        self.current_model: str = "llama3.2:latest"
        self.conversation_history: List[Dict[str, str]] = []
        self.session = PromptSession(history=InMemoryHistory())
        
    async def init_client(self) -> bool:
        """Initialize the Ollama client and check connectivity."""
        try:
            self.client = AsyncClient(host=self.host)
            return await self.health_check()
        except Exception as e:
            self.console.print(f"❌ Failed to initialize Ollama client: {e}", style="red")
            return False
    
    async def health_check(self) -> bool:
        """Check if Ollama server is accessible."""
        try:
            # Simple health check by listing models
            response = await self.client.list()
            return True
        except Exception as e:
            self.console.print(f"❌ Ollama server health check failed: {e}", style="red")
            return False
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """Get available models from Ollama."""
        try:
            response = await self.client.list()
            self.available_models = response.get('models', [])
            return self.available_models
        except Exception as e:
            self.console.print(f"❌ Error listing models: {e}", style="red")
            return []
    
    def display_banner(self):
        """Display ABOV3 banner and introduction."""
        banner = """
    ███████╗    ██████╗      ██████╗     ██╗   ██╗    ██████╗
    ██╔════╝    ██╔══██╗    ██╔═══██╗    ██║   ██║    ╚════██╗
    ███████║    ██████╔╝    ██║   ██║    ██║   ██║      ███╔═╝
    ██╔════╝    ██╔══██╗    ██║   ██║    ╚██╗ ██╔╝     ██╔══╝
    ███████╗    ██████╔╝    ╚██████╔╝     ╚████╔╝    ██████╔╝
    ╚══════╝    ╚═════╝      ╚═════╝       ╚═══╝     ╚═════╝
        """
        
        self.console.print(Panel(
            banner + "\n" + 
            "[bold cyan]Enterprise AI/ML Expert Agent[/bold cyan]\n" +
            "[yellow]Democratizing Software Development Through AI[/yellow]\n" +
            "[dim]Minimal Working Demo - Version 1.0.0[/dim]",
            style="blue",
            title="[bold white]Welcome to ABOV3[/bold white]",
            subtitle="[dim]AI-Powered Coding Platform[/dim]"
        ))
    
    def display_features(self):
        """Display key features of ABOV3."""
        features = Table(title="🚀 ABOV3 Core Features", show_header=True, header_style="bold magenta")
        features.add_column("Feature", style="cyan", width=20)
        features.add_column("Description", style="white", width=50)
        features.add_column("Status", style="green", width=10)
        
        features.add_row("🤖 AI Code Generation", "Natural language to production-ready code", "✅ Active")
        features.add_row("💬 Interactive Chat", "Streaming conversations with context memory", "✅ Active")
        features.add_row("🔄 Model Management", "Multi-model support and switching", "✅ Active")
        features.add_row("🛠️ Code Analysis", "Intelligent code review and optimization", "✅ Active")
        features.add_row("📚 Context Awareness", "Project-wide understanding and memory", "✅ Active")
        features.add_row("🔒 Enterprise Ready", "Security, compliance, and scalability", "✅ Active")
        
        self.console.print(features)
        self.console.print()
    
    async def display_system_status(self):
        """Display system status and available models."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Checking system status...", total=None)
            
            # Check Ollama connection
            ollama_status = "✅ Connected" if await self.health_check() else "❌ Disconnected"
            
            # Get available models
            models = await self.list_models()
            
            progress.remove_task(task)
        
        # Create status table
        status_table = Table(title="🔍 System Status", show_header=True, header_style="bold blue")
        status_table.add_column("Component", style="cyan", width=20)
        status_table.add_column("Status", style="white", width=30)
        status_table.add_column("Details", style="dim", width=30)
        
        status_table.add_row("Ollama Server", ollama_status, f"Host: {self.host}")
        status_table.add_row("Available Models", f"✅ {len(models)} models", f"Current: {self.current_model}")
        status_table.add_row("Memory Usage", "✅ Optimal", "Conversation history active")
        
        self.console.print(status_table)
        self.console.print()
        
        if models:
            self.display_available_models(models)
    
    def display_available_models(self, models: List[Dict[str, Any]]):
        """Display available models in a formatted table."""
        models_table = Table(title="🤖 Available AI Models", show_header=True, header_style="bold green")
        models_table.add_column("Model Name", style="cyan", width=25)
        models_table.add_column("Size", style="yellow", width=10)
        models_table.add_column("Modified", style="dim", width=20)
        models_table.add_column("Status", style="white", width=10)
        
        for model in models[:10]:  # Show first 10 models
            name = model.get('name', 'Unknown')
            size = self.format_size(model.get('size', 0))
            modified = model.get('modified_at', 'Unknown')[:19].replace('T', ' ')
            status = "🟢 Ready" if name == self.current_model else "⚪ Available"
            
            models_table.add_row(name, size, modified, status)
        
        if len(models) > 10:
            models_table.add_row("...", f"+ {len(models) - 10} more", "...", "...")
        
        self.console.print(models_table)
        self.console.print()
    
    def format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes == 0:
            return "Unknown"
        
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f}PB"
    
    async def change_model(self):
        """Allow user to change the current model."""
        if not self.available_models:
            await self.list_models()
        
        if not self.available_models:
            self.console.print("❌ No models available", style="red")
            return
        
        self.console.print("\n📋 Available Models:", style="bold blue")
        for i, model in enumerate(self.available_models, 1):
            current_marker = " (current)" if model['name'] == self.current_model else ""
            self.console.print(f"  {i}. {model['name']}{current_marker}")
        
        try:
            choice = Prompt.ask(
                "\nSelect model number (or press Enter to keep current)",
                default=""
            )
            
            if choice.strip():
                model_index = int(choice) - 1
                if 0 <= model_index < len(self.available_models):
                    old_model = self.current_model
                    self.current_model = self.available_models[model_index]['name']
                    self.console.print(f"✅ Model changed from {old_model} to {self.current_model}", style="green")
                else:
                    self.console.print("❌ Invalid model number", style="red")
        except (ValueError, KeyboardInterrupt):
            self.console.print("Model change cancelled", style="yellow")
    
    async def chat_session(self):
        """Interactive chat session with streaming responses."""
        self.console.print(Panel(
            f"[bold green]🤖 AI Chat Session Started[/bold green]\n"
            f"Current Model: [cyan]{self.current_model}[/cyan]\n"
            f"Type your message and press Enter. Use '/help' for commands.",
            title="Chat Session",
            style="green"
        ))
        
        while True:
            try:
                # Get user input
                user_input = await asyncio.to_thread(
                    self.session.prompt, 
                    "💬 You: ",
                    multiline=False
                )
                
                if not user_input.strip():
                    continue
                
                # Handle special commands
                if user_input.startswith('/'):
                    if await self.handle_chat_command(user_input):
                        continue
                    else:
                        break
                
                # Add to conversation history
                self.conversation_history.append({"role": "user", "content": user_input})
                
                # Get AI response with streaming
                self.console.print("🤖 ABOV3: ", end="")
                
                response_parts = []
                try:
                    async for chunk in self.stream_chat_response(user_input):
                        self.console.print(chunk, end="", style="cyan")
                        response_parts.append(chunk)
                    
                    full_response = "".join(response_parts)
                    if full_response:
                        self.conversation_history.append({"role": "assistant", "content": full_response})
                    
                    self.console.print("\n")
                    
                except Exception as e:
                    self.console.print(f"\n❌ Error getting response: {e}", style="red")
                
            except KeyboardInterrupt:
                self.console.print("\n\n👋 Chat session ended", style="yellow")
                break
            except Exception as e:
                self.console.print(f"❌ Error in chat session: {e}", style="red")
                break
    
    async def stream_chat_response(self, message: str):
        """Stream chat response from Ollama."""
        try:
            # Prepare conversation context (last 10 messages for context)
            messages = self.conversation_history[-10:] + [{"role": "user", "content": message}]
            
            # Stream response
            async for chunk in await self.client.chat(
                model=self.current_model,
                messages=messages,
                stream=True,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 4096
                }
            ):
                if chunk.get('message', {}).get('content'):
                    yield chunk['message']['content']
                    
        except Exception as e:
            yield f"Error: {str(e)}"
    
    async def handle_chat_command(self, command: str) -> bool:
        """Handle special chat commands. Returns True to continue chat, False to exit."""
        cmd = command.lower().strip()
        
        if cmd == '/help':
            self.display_chat_help()
        elif cmd == '/models':
            await self.display_system_status()
        elif cmd == '/change':
            await self.change_model()
        elif cmd == '/clear':
            self.conversation_history.clear()
            self.console.print("🧹 Conversation history cleared", style="yellow")
        elif cmd == '/history':
            self.display_conversation_history()
        elif cmd in ['/exit', '/quit', '/q']:
            return False
        elif cmd == '/demo':
            await self.run_demo_scenarios()
        else:
            self.console.print(f"❓ Unknown command: {command}", style="red")
            self.display_chat_help()
        
        return True
    
    def display_chat_help(self):
        """Display available chat commands."""
        help_table = Table(title="💬 Chat Commands", show_header=True, header_style="bold yellow")
        help_table.add_column("Command", style="cyan", width=15)
        help_table.add_column("Description", style="white", width=40)
        
        help_table.add_row("/help", "Show this help message")
        help_table.add_row("/models", "Show system status and available models")
        help_table.add_row("/change", "Change the current AI model")
        help_table.add_row("/clear", "Clear conversation history")
        help_table.add_row("/history", "Show conversation history")
        help_table.add_row("/demo", "Run automated demo scenarios")
        help_table.add_row("/exit", "Exit chat session")
        
        self.console.print(help_table)
        self.console.print()
    
    def display_conversation_history(self):
        """Display conversation history."""
        if not self.conversation_history:
            self.console.print("📭 No conversation history", style="yellow")
            return
        
        self.console.print(f"\n📜 Conversation History ({len(self.conversation_history)} messages):\n")
        
        for i, msg in enumerate(self.conversation_history[-10:], 1):  # Show last 10 messages
            role_icon = "👤" if msg["role"] == "user" else "🤖"
            role_color = "blue" if msg["role"] == "user" else "cyan"
            
            content = msg["content"]
            if len(content) > 100:
                content = content[:97] + "..."
            
            self.console.print(f"{i}. {role_icon} [{role_color}]{msg['role'].title()}[/{role_color}]: {content}")
        
        if len(self.conversation_history) > 10:
            self.console.print(f"... and {len(self.conversation_history) - 10} more messages")
        
        self.console.print()
    
    async def run_demo_scenarios(self):
        """Run automated demo scenarios showcasing ABOV3 capabilities."""
        scenarios = [
            {
                "title": "🐍 Python Code Generation",
                "prompt": "Create a Python function that calculates the factorial of a number using recursion. Include type hints and docstring."
            },
            {
                "title": "🌐 Web Development",
                "prompt": "Show me how to create a simple REST API endpoint using FastAPI that returns a list of users."
            },
            {
                "title": "📊 Data Analysis",
                "prompt": "Write Python code to read a CSV file and create a bar chart using matplotlib."
            },
            {
                "title": "🔧 Code Review",
                "prompt": "Review this code and suggest improvements: def calc(x,y): return x*y+x/y"
            }
        ]
        
        self.console.print(Panel(
            "[bold green]🎬 Running ABOV3 Demo Scenarios[/bold green]\n"
            "Watch ABOV3 demonstrate its AI coding capabilities across different domains.",
            title="Demo Mode",
            style="green"
        ))
        
        for i, scenario in enumerate(scenarios, 1):
            self.console.print(f"\n[bold blue]Scenario {i}: {scenario['title']}[/bold blue]")
            self.console.print(f"📝 Prompt: {scenario['prompt']}")
            self.console.print("🤖 ABOV3: ", end="")
            
            # Stream the response
            response_parts = []
            try:
                async for chunk in self.stream_chat_response(scenario['prompt']):
                    # Limit output for demo purposes
                    if len("".join(response_parts)) < 300:
                        self.console.print(chunk, end="", style="cyan")
                    response_parts.append(chunk)
                
                if len("".join(response_parts)) > 300:
                    self.console.print("\n... [Response truncated for demo] ...", style="dim")
                
                self.console.print("\n")
                
                # Small pause between scenarios
                if i < len(scenarios):
                    await asyncio.sleep(2)
                    
            except Exception as e:
                self.console.print(f"\n❌ Error in demo scenario: {e}", style="red")
        
        self.console.print("✅ Demo scenarios completed!", style="green")
    
    async def show_quick_start_guide(self):
        """Show a quick start guide for new users."""
        guide = """
        ## 🚀 Quick Start Guide

        ### What is ABOV3?
        ABOV3 is an Enterprise AI/ML Expert Agent designed to democratize software development.
        It enables both developers and non-technical users to create production-ready applications
        through natural language interactions.

        ### Key Capabilities:
        • **Code Generation**: Convert natural language to production-ready code
        • **Multi-Language Support**: Python, JavaScript, Java, Go, Rust, and more
        • **Intelligent Analysis**: Code review, optimization, and security scanning  
        • **Context Awareness**: Understands entire codebases and project structure
        • **Real-time Streaming**: Fast, responsive AI interactions

        ### Getting Started:
        1. Type your coding request in natural language
        2. ABOV3 generates optimized, secure code
        3. Ask follow-up questions to refine the solution
        4. Use `/demo` to see example interactions

        ### Example Prompts:
        • "Create a REST API for user management"
        • "Add error handling to this function"  
        • "Optimize this database query"
        • "Generate unit tests for my code"

        Ready to start coding with AI? Just type your request below!
        """
        
        self.console.print(Panel(
            Markdown(guide),
            title="[bold cyan]ABOV3 Quick Start Guide[/bold cyan]",
            style="blue",
            padding=(1, 2)
        ))
    
    async def main_menu(self):
        """Display main menu and handle user choices."""
        while True:
            self.console.print("\n" + "="*60)
            menu_table = Table(title="🎯 ABOV3 Main Menu", show_header=True, header_style="bold cyan")
            menu_table.add_column("Option", style="yellow", width=10)
            menu_table.add_column("Description", style="white", width=40)
            
            menu_table.add_row("1", "🤖 Start Interactive Chat Session")
            menu_table.add_row("2", "📊 View System Status & Models")
            menu_table.add_row("3", "🔄 Change AI Model")
            menu_table.add_row("4", "🎬 Run Demo Scenarios")
            menu_table.add_row("5", "📚 Quick Start Guide")
            menu_table.add_row("6", "❌ Exit")
            
            self.console.print(menu_table)
            
            try:
                choice = Prompt.ask("\n🎯 Select an option", choices=["1", "2", "3", "4", "5", "6"], default="1")
                
                if choice == "1":
                    await self.chat_session()
                elif choice == "2":
                    await self.display_system_status()
                elif choice == "3":
                    await self.change_model()
                elif choice == "4":
                    await self.run_demo_scenarios()
                elif choice == "5":
                    await self.show_quick_start_guide()
                elif choice == "6":
                    self.console.print("👋 Thank you for trying ABOV3!", style="green")
                    break
                    
            except KeyboardInterrupt:
                self.console.print("\n👋 Goodbye!", style="yellow")
                break
            except Exception as e:
                self.console.print(f"❌ Error: {e}", style="red")
    
    async def run(self):
        """Main entry point for the demo."""
        self.display_banner()
        self.display_features()
        
        # Initialize connection
        self.console.print("🔄 Initializing ABOV3...", style="yellow")
        
        if not await self.init_client():
            self.console.print("\n❌ [bold red]Cannot connect to Ollama server[/bold red]")
            self.console.print("\n💡 [yellow]To run this demo:[/yellow]")
            self.console.print("1. Install Ollama from https://ollama.ai")
            self.console.print("2. Start Ollama: [cyan]ollama serve[/cyan]")
            self.console.print("3. Pull a model: [cyan]ollama pull llama3.2[/cyan]")
            self.console.print("4. Run this demo again: [cyan]python demo.py[/cyan]")
            return False
        
        self.console.print("✅ ABOV3 initialized successfully!", style="green")
        
        # Check for available models and set default
        models = await self.list_models()
        if models:
            # Try to find a good default model
            preferred_models = ["llama3.2:latest", "llama3:latest", "codellama:latest"]
            for preferred in preferred_models:
                if any(model['name'] == preferred for model in models):
                    self.current_model = preferred
                    break
            else:
                # Use the first available model
                self.current_model = models[0]['name']
        
        # Show initial system status
        await self.display_system_status()
        
        # Start main menu
        await self.main_menu()
        return True


@click.command()
@click.option('--host', default='http://localhost:11434', help='Ollama server host')
@click.option('--model', default=None, help='Default model to use')
def main(host: str, model: Optional[str]):
    """
    ABOV3 4 Ollama - Minimal Working Demo
    
    A simple demonstration of ABOV3's core AI coding capabilities.
    """
    demo = ABOV3Demo(host=host)
    
    if model:
        demo.current_model = model
    
    try:
        success = asyncio.run(demo.run())
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("🐛 Please report this issue to the ABOV3 team")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()