#!/usr/bin/env python3
"""
Trace Inspector Tool for SmolagentsBenchmark Output
A tool for interactively browsing and analyzing agent conversation traces.
"""

import json
import os
from typing import List, Dict, Any
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.syntax import Syntax
import re

console = Console()

class TraceInspector:
    def __init__(self, trace_file: str):
        """Initialize the trace inspector with a single trace file."""
        self.trace_file = trace_file
        self.traces: List[Dict[str, Any]] = []
        self.current_trace_idx = 0
        self.current_step_idx = 0
        self.show_tool_calls = False  # Toggle for showing tool calls
        
    def load_file(self):
        """Load traces from the file."""
        if not os.path.exists(self.trace_file):
            console.print(f"[red]File not found: {self.trace_file}[/red]")
            return False
            
        self.traces = []
        
        with open(self.trace_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        trace = json.loads(line.strip())
                        self.traces.append(trace)
                    except json.JSONDecodeError as e:
                        console.print(f"[yellow]Skipping invalid JSON line: {e}[/yellow]")
                        continue
                    
        console.print(f"[cyan]Loaded {len(self.traces)} traces from {os.path.basename(self.trace_file)}[/cyan]")
        self.current_trace_idx = 0
        self.current_step_idx = 0
        return len(self.traces) > 0
        
    def display_trace_info(self):
        """Display current trace information."""
        if not self.traces:
            return
            
        trace = self.traces[self.current_trace_idx]
        
        # Create info table
        table = Table(title=f"Trace {self.current_trace_idx + 1}/{len(self.traces)}")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")
        
        # Basic info
        table.add_row("Model ID", trace.get("model_id", "N/A"))
        table.add_row("Action Type", trace.get("agent_action_type", "N/A"))
        table.add_row("Source", trace.get("source", "N/A"))
        table.add_row("Answer", str(trace.get("answer", "N/A")))
        table.add_row("True Answer", str(trace.get("true_answer", "N/A")))
        
        if "intermediate_steps" in trace:
            table.add_row("Total Steps", str(len(trace["intermediate_steps"])))
            
        console.print(table)
        
        # Display question
        if "question" in trace:
            question_panel = Panel(
                trace["question"],
                title="[bold]Question[/bold]",
                border_style="blue"
            )
            console.print(question_panel)
            
    def get_filtered_steps(self):
        """Get steps filtered based on show_tool_calls setting."""
        if not self.traces or "intermediate_steps" not in self.traces[self.current_trace_idx]:
            return []
        
        steps = self.traces[self.current_trace_idx]["intermediate_steps"]
        
        if self.show_tool_calls:
            return steps
        else:
            # Filter out tool-call and tool-response steps
            return [s for s in steps if s.get("role") not in ["tool-call", "tool-response"]]
    
    def display_step(self):
        """Display current conversation step."""
        if not self.traces:
            return
            
        steps = self.get_filtered_steps()
        
        if not steps:
            console.print("[yellow]No steps in this trace[/yellow]")
            return
            
        if self.current_step_idx >= len(steps):
            console.print("[yellow]No more steps[/yellow]")
            return
            
        step = steps[self.current_step_idx]
        
        # Display step header
        header = f"Step {self.current_step_idx + 1}/{len(steps)} - Role: {step.get('role', 'unknown')}"
        console.print(f"\n[bold magenta]{header}[/bold magenta]")
        console.print("=" * len(header))
        
        # Display content based on role
        if step.get("role") == "assistant":
            self._display_assistant_step(step)
            # If next steps are tool calls, show them inline
            if not self.show_tool_calls:
                self._display_related_tool_calls(self.current_step_idx)
        elif step.get("role") == "user":
            self._display_user_step(step)
        elif step.get("role") == "tool-call":
            self._display_tool_call(step)
        elif step.get("role") == "tool-response":
            self._display_tool_response(step)
        elif step.get("role") == "system":
            self._display_system_step(step)
            
    def _display_assistant_step(self, step: Dict[str, Any]):
        """Display assistant response step."""
        content = step.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    
                    # Extract and display thought/code blocks specially
                    
                    parts = re.split(r"(Thought:|<code>|</code>)", text)
                    
                    current_mode = "text"
                    for part in parts:
                        if part == "Thought:":
                            current_mode = "thought"
                            console.print("\n[bold yellow]Thought:[/bold yellow]")
                        elif part == "<code>":
                            current_mode = "code"
                        elif part == "</code>":
                            current_mode = "text"
                        elif part.strip():
                            if current_mode == "code":
                                syntax = Syntax(part.strip(), "python", theme="monokai", line_numbers=True)
                                console.print(Panel(syntax, title="Code", border_style="green"))
                            elif current_mode == "thought":
                                console.print(Text(part.strip(), style="yellow"))
                            else:
                                console.print(part.strip())
                                
    def _display_user_step(self, step: Dict[str, Any]):
        """Display user message step."""
        content = step.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    panel = Panel(text, title="[bold]User Input[/bold]", border_style="cyan")
                    console.print(panel)
                    
    def _display_tool_call(self, step: Dict[str, Any]):
        """Display tool call step."""
        content = step.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    # Parse tool calls
                    console.print(Panel(text, title="[bold]Tool Call[/bold]", border_style="magenta"))
                    
    def _display_tool_response(self, step: Dict[str, Any]):
        """Display tool response step."""
        content = step.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    # Check if it's an observation
                    if text.startswith("Observation:"):
                        obs_text = text.replace("Observation:", "", 1).strip()
                        panel = Panel(obs_text, title="[bold]Observation[/bold]", border_style="green")
                        console.print(panel)
                    else:
                        console.print(Panel(text, title="[bold]Tool Response[/bold]", border_style="green"))
                        
    def _display_system_step(self, step: Dict[str, Any]):
        """Display system message step."""
        content = step.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    # Truncate very long system messages
                    if len(text) > 1000:
                        text = text[:1000] + "\n... (truncated)"
                    panel = Panel(text, title="[bold]System[/bold]", border_style="red")
                    console.print(panel)
                    
    def _display_related_tool_calls(self, assistant_step_idx: int):
        """Display tool calls and responses related to an assistant step."""
        if not self.traces or "intermediate_steps" not in self.traces[self.current_trace_idx]:
            return
            
        all_steps = self.traces[self.current_trace_idx]["intermediate_steps"]
        
        # Find the actual index in all_steps
        actual_idx = -1
        count = 0
        for i, step in enumerate(all_steps):
            if step.get("role") not in ["tool-call", "tool-response"]:
                if count == assistant_step_idx:
                    actual_idx = i
                    break
                count += 1
        
        if actual_idx == -1:
            return
            
        # Look for tool calls immediately after this assistant step
        i = actual_idx + 1
        while i < len(all_steps):
            step = all_steps[i]
            if step.get("role") == "tool-call":
                console.print("\n[dim]→ Tool Call[/dim]")
                self._display_tool_call(step)
            elif step.get("role") == "tool-response":
                console.print("[dim]→ Tool Response[/dim]")
                self._display_tool_response(step)
            else:
                break  # Stop when we hit a non-tool step
            i += 1
                    
    def run_interactive(self):
        """Run interactive trace browser."""
        if not self.load_file():
            return
        
        console.print("\n[bold green]Trace Inspector[/bold green]")
        console.print("Commands: n=next step, p=prev step, N=next trace, P=prev trace")
        console.print("         i=info, r=reset, j=jump, t=toggle tool calls, q=quit")
        console.print("-" * 60)
        
        self.display_trace_info()
        
        while True:
            try:
                # Display current position
                status = f"\nTrace [{self.current_trace_idx + 1}/{len(self.traces)}] | "
                
                steps = self.get_filtered_steps()
                if steps:
                    status += f"Step [{self.current_step_idx + 1}/{len(steps)}]"
                
                status += f" | Tool calls: {'shown' if self.show_tool_calls else 'hidden'}"
                    
                console.print(status, style="bold blue")
                
                cmd = input("\nCommand: ").strip()
                
                if cmd.lower() == "q":
                    break
                elif cmd == "n":  # Next step
                    steps = self.get_filtered_steps()
                    if steps:
                        if self.current_step_idx < len(steps) - 1:
                            self.current_step_idx += 1
                            self.display_step()
                        else:
                            console.print("[yellow]Already at last step[/yellow]")
                    else:
                        console.print("[yellow]No steps in this trace[/yellow]")
                elif cmd == "p":  # Previous step
                    if self.current_step_idx > 0:
                        self.current_step_idx -= 1
                        self.display_step()
                    else:
                        console.print("[yellow]Already at first step[/yellow]")
                elif cmd == "N":  # Next trace
                    if self.current_trace_idx < len(self.traces) - 1:
                        self.current_trace_idx += 1
                        self.current_step_idx = 0
                        console.clear()
                        self.display_trace_info()
                    else:
                        console.print("[yellow]Already at last trace[/yellow]")
                elif cmd == "P":  # Previous trace
                    if self.current_trace_idx > 0:
                        self.current_trace_idx -= 1
                        self.current_step_idx = 0
                        console.clear()
                        self.display_trace_info()
                    else:
                        console.print("[yellow]Already at first trace[/yellow]")
                elif cmd == "i":  # Show info
                    console.clear()
                    self.display_trace_info()
                elif cmd == "r":  # Reset to first step
                    self.current_step_idx = 0
                    console.clear()
                    self.display_trace_info()
                elif cmd == "j":  # Jump to trace
                    try:
                        trace_num = int(input("Enter trace number: ")) - 1
                        if 0 <= trace_num < len(self.traces):
                            self.current_trace_idx = trace_num
                            self.current_step_idx = 0
                            console.clear()
                            self.display_trace_info()
                        else:
                            console.print("[red]Invalid trace number[/red]")
                    except ValueError:
                        console.print("[red]Invalid input[/red]")
                elif cmd == "t":  # Toggle tool calls
                    self.show_tool_calls = not self.show_tool_calls
                    self.current_step_idx = 0  # Reset to first step after toggle
                    console.clear()
                    console.print(f"[cyan]Tool calls are now {'shown' if self.show_tool_calls else 'hidden'}[/cyan]")
                    self.display_trace_info()
                elif cmd.isdigit():  # Jump to trace number
                    trace_num = int(cmd) - 1
                    if 0 <= trace_num < len(self.traces):
                        self.current_trace_idx = trace_num
                        self.current_step_idx = 0
                        console.clear()
                        self.display_trace_info()
                    else:
                        console.print("[red]Invalid trace number[/red]")
                elif cmd == "":  # Enter to show current step
                    self.display_step()
                else:
                    console.print("[yellow]Unknown command. Use h for help.[/yellow]")
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'q' to quit[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                
        console.print("\n[green]Goodbye![/green]")

def main():
    parser = argparse.ArgumentParser(description="Inspect agent conversation traces")
    parser.add_argument(
        "file",
        nargs='?',
        default="./output/qwen2.5-coder-7b-math-lr2e-5-len8192-epoch2-verif0.2-filtered-assis-only-trigger-stc__code__math__2025-08-26.jsonl",
        help="Path to trace file (.jsonl)"
    )
    
    args = parser.parse_args()
    
    inspector = TraceInspector(trace_file=args.file)
    inspector.run_interactive()

if __name__ == "__main__":
    main()