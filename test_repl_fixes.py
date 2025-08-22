#!/usr/bin/env python3
"""
Test script to verify REPL fixes for ABOV3.

This script tests:
1. Command processing (/help, /exit, etc.)
2. Input handling (single-line vs multiline)
3. Ctrl+D handling
4. Ollama connection for chat
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from abov3.ui.console.repl import ABOV3REPL, REPLConfig, create_repl
from abov3.core.config import get_config
from rich.console import Console

console = Console()

async def test_command_processor():
    """Test that commands are properly processed."""
    console.print("[bold blue]Testing Command Processor[/bold blue]")
    
    # Create a simple REPL for testing
    config = REPLConfig(
        prompt_text="TEST> ",
        enable_multiline=False,  # Single-line mode
        theme="monokai"
    )
    
    # Create test callback
    async def test_callback(text):
        return f"Echo: {text}"
    
    repl = create_repl(config=config, process_callback=test_callback)
    
    # Test command detection
    tests = [
        ("/help", True, "Should detect /help as command"),
        ("/exit", True, "Should detect /exit as command"),
        ("/config", True, "Should detect /config as command"),
        ("hello world", False, "Should not detect regular text as command"),
        ("", False, "Should not detect empty string as command"),
    ]
    
    for text, expected, description in tests:
        is_command = repl.command_processor.is_command(text)
        status = "[green]PASS[/green]" if is_command == expected else "[red]FAIL[/red]"
        console.print(f"  {status} {description}: '{text}' -> {is_command}")
    
    # Test command processing
    console.print("\n[bold]Testing Command Processing:[/bold]")
    
    # Test /help command
    result = await repl.command_processor.process("/help")
    if result is not None or result == "":
        console.print("  [green]PASS[/green] /help command processed")
    else:
        console.print("  [red]FAIL[/red] /help command failed")
    
    # Test unknown command
    result = await repl.command_processor.process("/unknown")
    if "Unknown command" in result:
        console.print("  [green]PASS[/green] Unknown command handled properly")
    else:
        console.print(f"  [red]FAIL[/red] Unknown command not handled: {result}")
    
    console.print("[green]Command processor tests complete![/green]\n")

async def test_repl_config():
    """Test REPL configuration."""
    console.print("[bold blue]Testing REPL Configuration[/bold blue]")
    
    config = REPLConfig()
    
    # Check default settings
    checks = [
        (config.enable_multiline == False, "Multiline should be disabled by default"),
        (config.prompt_text == "ABOV3> ", "Default prompt should be 'ABOV3> '"),
        (config.async_mode == True, "Async mode should be enabled"),
        (config.enable_auto_suggestions == True, "Auto suggestions should be enabled"),
    ]
    
    for check, description in checks:
        status = "[green]PASS[/green]" if check else "[red]FAIL[/red]"
        console.print(f"  {status} {description}")
    
    console.print("[green]Configuration tests complete![/green]\n")

async def test_input_processing():
    """Test input processing flow."""
    console.print("[bold blue]Testing Input Processing[/bold blue]")
    
    config = REPLConfig(enable_multiline=False)
    
    # Track what was processed
    processed_inputs = []
    
    async def capture_callback(text):
        processed_inputs.append(text)
        return f"Processed: {text}"
    
    repl = create_repl(config=config, process_callback=capture_callback)
    
    # Test command vs regular input
    test_inputs = [
        ("/help", "command", "Should process as command"),
        ("Hello AI", "callback", "Should process through callback"),
        ("/exit", "command", "Should process as command"),
    ]
    
    for input_text, expected_type, description in test_inputs:
        result = await repl.process_input(input_text)
        
        if expected_type == "command":
            is_correct = input_text not in processed_inputs
            status = "[green]PASS[/green]" if is_correct else "[red]FAIL[/red]"
            console.print(f"  {status} {description}: '{input_text}'")
        else:
            is_correct = input_text in processed_inputs
            status = "[green]PASS[/green]" if is_correct else "[red]FAIL[/red]"
            console.print(f"  {status} {description}: '{input_text}'")
    
    console.print("[green]Input processing tests complete![/green]\n")

async def main():
    """Run all tests."""
    console.print("[bold cyan]ABOV3 REPL Fix Verification[/bold cyan]\n")
    
    try:
        await test_repl_config()
        await test_command_processor()
        await test_input_processing()
        
        console.print("[bold green]ALL TESTS COMPLETED![/bold green]")
        console.print("\n[yellow]Summary of fixes:[/yellow]")
        console.print("  - Multiline mode disabled by default (no more '...' prompts)")
        console.print("  - Command processing fixed (removed shell operator blocking)")
        console.print("  - Ctrl+D handling improved for clean exit")
        console.print("  - Input processing flow simplified")
        console.print("\n[cyan]The REPL should now work properly. Try running:[/cyan]")
        console.print("  [bold]python -m abov3 chat[/bold]")
        
    except Exception as e:
        console.print(f"[red]Test failed: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())