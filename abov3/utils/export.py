"""
Export Utilities for ABOV3 4 Ollama.

This module provides comprehensive export capabilities for conversations,
including multiple formats, code extraction, and custom templates.
"""

import asyncio
import base64
import json
import logging
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from ..core.history.manager import HistoryManager, ConversationThread, Message, MessageType
from ..core.config import get_config
from .security import SecurityManager


logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Export format enumeration."""
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    TXT = "txt"
    PDF = "pdf"
    CSV = "csv"
    XML = "xml"
    YAML = "yaml"


class CodeLanguage(Enum):
    """Supported code languages for extraction."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    C = "c"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    SHELL = "shell"
    SQL = "sql"
    HTML = "html"
    CSS = "css"
    YAML = "yaml"
    JSON = "json"
    XML = "xml"
    OTHER = "other"


@dataclass
class ExportOptions:
    """Export configuration options."""
    
    format: ExportFormat = ExportFormat.JSON
    include_metadata: bool = True
    include_timestamps: bool = True
    include_tokens: bool = False
    include_model_info: bool = True
    include_context: bool = False
    extract_code: bool = False
    code_languages: List[CodeLanguage] = field(default_factory=list)
    template_path: Optional[Path] = None
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    compress_output: bool = False
    sanitize_content: bool = True
    max_file_size_mb: int = 100


@dataclass
class CodeBlock:
    """Extracted code block."""
    
    language: CodeLanguage
    content: str
    source_message_id: str
    line_start: int = 0
    line_end: int = 0
    filename: Optional[str] = None
    description: Optional[str] = None


@dataclass
class ExportResult:
    """Export operation result."""
    
    success: bool
    output_path: Optional[Path] = None
    output_content: Optional[str] = None
    file_size_bytes: int = 0
    exported_conversations: int = 0
    exported_messages: int = 0
    extracted_code_blocks: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationExporter:
    """
    Comprehensive conversation export system.
    
    Features:
    - Multiple export formats (JSON, Markdown, HTML, PDF, etc.)
    - Code block extraction and organization
    - Template-based exports
    - Batch export operations
    - Content sanitization
    - Custom export formats
    - Async support for large exports
    """
    
    def __init__(self, history_manager: HistoryManager, config: Optional[Dict[str, Any]] = None):
        """Initialize the conversation exporter."""
        self.history_manager = history_manager
        self.config = get_config()
        self._custom_config = config or {}
        
        # Security
        self.security_manager = SecurityManager()
        
        # Code extraction patterns
        self.code_patterns = {
            CodeLanguage.PYTHON: [
                r'```python\n(.*?)\n```',
                r'```py\n(.*?)\n```',
                r'`([^`\n]*\.py[^`\n]*)`'
            ],
            CodeLanguage.JAVASCRIPT: [
                r'```javascript\n(.*?)\n```',
                r'```js\n(.*?)\n```',
                r'`([^`\n]*\.js[^`\n]*)`'
            ],
            CodeLanguage.TYPESCRIPT: [
                r'```typescript\n(.*?)\n```',
                r'```ts\n(.*?)\n```',
                r'`([^`\n]*\.ts[^`\n]*)`'
            ],
            CodeLanguage.JAVA: [
                r'```java\n(.*?)\n```',
                r'`([^`\n]*\.java[^`\n]*)`'
            ],
            CodeLanguage.CSHARP: [
                r'```csharp\n(.*?)\n```',
                r'```cs\n(.*?)\n```',
                r'`([^`\n]*\.cs[^`\n]*)`'
            ],
            CodeLanguage.CPP: [
                r'```cpp\n(.*?)\n```',
                r'```c\+\+\n(.*?)\n```',
                r'`([^`\n]*\.cpp[^`\n]*)`',
                r'`([^`\n]*\.hpp[^`\n]*)`'
            ],
            CodeLanguage.C: [
                r'```c\n(.*?)\n```',
                r'`([^`\n]*\.c[^`\n]*)`',
                r'`([^`\n]*\.h[^`\n]*)`'
            ],
            CodeLanguage.GO: [
                r'```go\n(.*?)\n```',
                r'`([^`\n]*\.go[^`\n]*)`'
            ],
            CodeLanguage.RUST: [
                r'```rust\n(.*?)\n```',
                r'```rs\n(.*?)\n```',
                r'`([^`\n]*\.rs[^`\n]*)`'
            ],
            CodeLanguage.PHP: [
                r'```php\n(.*?)\n```',
                r'`([^`\n]*\.php[^`\n]*)`'
            ],
            CodeLanguage.RUBY: [
                r'```ruby\n(.*?)\n```',
                r'```rb\n(.*?)\n```',
                r'`([^`\n]*\.rb[^`\n]*)`'
            ],
            CodeLanguage.SHELL: [
                r'```bash\n(.*?)\n```',
                r'```sh\n(.*?)\n```',
                r'```shell\n(.*?)\n```',
                r'`([^`\n]*\.sh[^`\n]*)`'
            ],
            CodeLanguage.SQL: [
                r'```sql\n(.*?)\n```',
                r'`([^`\n]*\.sql[^`\n]*)`'
            ],
            CodeLanguage.HTML: [
                r'```html\n(.*?)\n```',
                r'`([^`\n]*\.html[^`\n]*)`'
            ],
            CodeLanguage.CSS: [
                r'```css\n(.*?)\n```',
                r'`([^`\n]*\.css[^`\n]*)`'
            ],
            CodeLanguage.YAML: [
                r'```yaml\n(.*?)\n```',
                r'```yml\n(.*?)\n```',
                r'`([^`\n]*\.ya?ml[^`\n]*)`'
            ],
            CodeLanguage.JSON: [
                r'```json\n(.*?)\n```',
                r'`([^`\n]*\.json[^`\n]*)`'
            ],
            CodeLanguage.XML: [
                r'```xml\n(.*?)\n```',
                r'`([^`\n]*\.xml[^`\n]*)`'
            ]
        }
        
        logger.info("ConversationExporter initialized")
    
    def export_conversation(
        self,
        conversation_id: str,
        options: ExportOptions,
        output_path: Optional[Path] = None
    ) -> ExportResult:
        """Export a single conversation."""
        
        try:
            # Get conversation and messages
            conversation = self.history_manager.get_conversation(conversation_id)
            if not conversation:
                return ExportResult(
                    success=False,
                    errors=[f"Conversation {conversation_id} not found"]
                )
            
            messages = self.history_manager.get_conversation_messages(conversation_id)
            
            # Export based on format
            if options.format == ExportFormat.JSON:
                result = self._export_json([conversation], [messages], options, output_path)
            elif options.format == ExportFormat.MARKDOWN:
                result = self._export_markdown([conversation], [messages], options, output_path)
            elif options.format == ExportFormat.HTML:
                result = self._export_html([conversation], [messages], options, output_path)
            elif options.format == ExportFormat.TXT:
                result = self._export_txt([conversation], [messages], options, output_path)
            elif options.format == ExportFormat.PDF:
                result = self._export_pdf([conversation], [messages], options, output_path)
            elif options.format == ExportFormat.CSV:
                result = self._export_csv([conversation], [messages], options, output_path)
            elif options.format == ExportFormat.XML:
                result = self._export_xml([conversation], [messages], options, output_path)
            elif options.format == ExportFormat.YAML:
                result = self._export_yaml([conversation], [messages], options, output_path)
            else:
                return ExportResult(
                    success=False,
                    errors=[f"Unsupported export format: {options.format}"]
                )
            
            result.exported_conversations = 1
            result.exported_messages = len(messages)
            
            # Extract code if requested
            if options.extract_code:
                code_blocks = self._extract_code_blocks(messages, options.code_languages)
                result.extracted_code_blocks = len(code_blocks)
                
                if code_blocks:
                    self._export_code_blocks(code_blocks, output_path, options)
            
            return result
        
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return ExportResult(
                success=False,
                errors=[f"Export failed: {str(e)}"]
            )
    
    def export_multiple_conversations(
        self,
        conversation_ids: List[str],
        options: ExportOptions,
        output_path: Optional[Path] = None
    ) -> ExportResult:
        """Export multiple conversations."""
        
        try:
            conversations = []
            all_messages = []
            
            for conv_id in conversation_ids:
                conversation = self.history_manager.get_conversation(conv_id)
                if conversation:
                    conversations.append(conversation)
                    messages = self.history_manager.get_conversation_messages(conv_id)
                    all_messages.append(messages)
            
            if not conversations:
                return ExportResult(
                    success=False,
                    errors=["No valid conversations found"]
                )
            
            # Export based on format
            if options.format == ExportFormat.JSON:
                result = self._export_json(conversations, all_messages, options, output_path)
            elif options.format == ExportFormat.MARKDOWN:
                result = self._export_markdown(conversations, all_messages, options, output_path)
            elif options.format == ExportFormat.HTML:
                result = self._export_html(conversations, all_messages, options, output_path)
            elif options.format == ExportFormat.TXT:
                result = self._export_txt(conversations, all_messages, options, output_path)
            elif options.format == ExportFormat.PDF:
                result = self._export_pdf(conversations, all_messages, options, output_path)
            elif options.format == ExportFormat.CSV:
                result = self._export_csv(conversations, all_messages, options, output_path)
            elif options.format == ExportFormat.XML:
                result = self._export_xml(conversations, all_messages, options, output_path)
            elif options.format == ExportFormat.YAML:
                result = self._export_yaml(conversations, all_messages, options, output_path)
            else:
                return ExportResult(
                    success=False,
                    errors=[f"Unsupported export format: {options.format}"]
                )
            
            result.exported_conversations = len(conversations)
            result.exported_messages = sum(len(messages) for messages in all_messages)
            
            # Extract code if requested
            if options.extract_code:
                all_flat_messages = [msg for messages in all_messages for msg in messages]
                code_blocks = self._extract_code_blocks(all_flat_messages, options.code_languages)
                result.extracted_code_blocks = len(code_blocks)
                
                if code_blocks:
                    self._export_code_blocks(code_blocks, output_path, options)
            
            return result
        
        except Exception as e:
            logger.error(f"Multi-export failed: {e}")
            return ExportResult(
                success=False,
                errors=[f"Multi-export failed: {str(e)}"]
            )
    
    def _export_json(
        self,
        conversations: List[ConversationThread],
        all_messages: List[List[Message]],
        options: ExportOptions,
        output_path: Optional[Path]
    ) -> ExportResult:
        """Export to JSON format."""
        
        data = {
            "export_info": {
                "format": "json",
                "version": "1.0",
                "exported_at": datetime.utcnow().isoformat(),
                "tool": "ABOV3 Ollama",
                "options": {
                    "include_metadata": options.include_metadata,
                    "include_timestamps": options.include_timestamps,
                    "include_tokens": options.include_tokens,
                    "include_model_info": options.include_model_info,
                    "include_context": options.include_context
                }
            },
            "conversations": []
        }
        
        for conversation, messages in zip(conversations, all_messages):
            conv_data = conversation.to_dict()
            
            # Filter fields based on options
            if not options.include_metadata:
                conv_data.pop("metadata", None)
            if not options.include_timestamps:
                conv_data.pop("created_at", None)
                conv_data.pop("updated_at", None)
            if not options.include_tokens:
                conv_data.pop("total_tokens", None)
            if not options.include_model_info:
                conv_data.pop("model_info", None)
            
            # Process messages
            conv_messages = []
            for message in messages:
                msg_data = message.to_dict()
                
                # Sanitize content if requested
                if options.sanitize_content:
                    msg_data["content"] = self._sanitize_content(msg_data["content"])
                
                # Filter fields based on options
                if not options.include_metadata:
                    msg_data.pop("metadata", None)
                if not options.include_timestamps:
                    msg_data.pop("timestamp", None)
                if not options.include_tokens:
                    msg_data.pop("token_count", None)
                if not options.include_model_info:
                    msg_data.pop("model_info", None)
                if not options.include_context:
                    msg_data.pop("context_used", None)
                
                conv_messages.append(msg_data)
            
            conv_data["messages"] = conv_messages
            data["conversations"].append(conv_data)
        
        # Add custom fields
        data["export_info"].update(options.custom_fields)
        
        # Generate output
        content = json.dumps(data, indent=2, ensure_ascii=False)
        
        if output_path:
            output_path = output_path.with_suffix('.json')
            output_path.write_text(content, encoding='utf-8')
            return ExportResult(
                success=True,
                output_path=output_path,
                file_size_bytes=output_path.stat().st_size
            )
        else:
            return ExportResult(
                success=True,
                output_content=content,
                file_size_bytes=len(content.encode('utf-8'))
            )
    
    def _export_markdown(
        self,
        conversations: List[ConversationThread],
        all_messages: List[List[Message]],
        options: ExportOptions,
        output_path: Optional[Path]
    ) -> ExportResult:
        """Export to Markdown format."""
        
        lines = []
        
        # Header
        lines.append("# ABOV3 Conversation Export")
        lines.append("")
        lines.append(f"**Exported:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"**Conversations:** {len(conversations)}")
        lines.append("")
        
        for conversation, messages in zip(conversations, all_messages):
            # Conversation header
            lines.append(f"## {conversation.title}")
            lines.append("")
            
            if conversation.description:
                lines.append(f"**Description:** {conversation.description}")
                lines.append("")
            
            if options.include_timestamps:
                lines.append(f"**Created:** {conversation.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                lines.append(f"**Updated:** {conversation.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
                lines.append("")
            
            if options.include_metadata and conversation.metadata:
                lines.append("**Metadata:**")
                for key, value in conversation.metadata.items():
                    lines.append(f"- {key}: {value}")
                lines.append("")
            
            if conversation.tags:
                lines.append(f"**Tags:** {', '.join(conversation.tags)}")
                lines.append("")
            
            # Messages
            lines.append("### Messages")
            lines.append("")
            
            for i, message in enumerate(messages, 1):
                lines.append(f"#### Message {i} ({message.type.value.title()})")
                
                if options.include_timestamps:
                    lines.append(f"*{message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}*")
                    lines.append("")
                
                # Content
                content = message.content
                if options.sanitize_content:
                    content = self._sanitize_content(content)
                
                lines.append(content)
                lines.append("")
                
                if options.include_tokens:
                    lines.append(f"*Tokens: {message.token_count}*")
                    lines.append("")
            
            lines.append("---")
            lines.append("")
        
        content = "\n".join(lines)
        
        if output_path:
            output_path = output_path.with_suffix('.md')
            output_path.write_text(content, encoding='utf-8')
            return ExportResult(
                success=True,
                output_path=output_path,
                file_size_bytes=output_path.stat().st_size
            )
        else:
            return ExportResult(
                success=True,
                output_content=content,
                file_size_bytes=len(content.encode('utf-8'))
            )
    
    def _export_html(
        self,
        conversations: List[ConversationThread],
        all_messages: List[List[Message]],
        options: ExportOptions,
        output_path: Optional[Path]
    ) -> ExportResult:
        """Export to HTML format."""
        
        html_parts = []
        
        # HTML header
        html_parts.append("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ABOV3 Conversation Export</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 2rem; line-height: 1.6; }
        .header { border-bottom: 2px solid #333; padding-bottom: 1rem; margin-bottom: 2rem; }
        .conversation { margin-bottom: 3rem; border: 1px solid #ddd; border-radius: 8px; padding: 1.5rem; }
        .conversation-title { color: #333; margin-top: 0; }
        .message { margin: 1rem 0; padding: 1rem; border-radius: 6px; }
        .message-user { background-color: #e3f2fd; }
        .message-assistant { background-color: #f3e5f5; }
        .message-system { background-color: #fff3e0; }
        .message-tool { background-color: #e8f5e8; }
        .message-error { background-color: #ffebee; }
        .message-header { font-weight: bold; margin-bottom: 0.5rem; }
        .timestamp { color: #666; font-size: 0.9em; }
        .metadata { background-color: #f5f5f5; padding: 0.5rem; border-radius: 4px; margin: 0.5rem 0; }
        pre { background-color: #f8f8f8; padding: 1rem; border-radius: 4px; overflow-x: auto; }
        code { background-color: #f0f0f0; padding: 0.2rem 0.4rem; border-radius: 3px; }
    </style>
</head>
<body>
""")
        
        # Header
        html_parts.append('<div class="header">')
        html_parts.append('<h1>ABOV3 Conversation Export</h1>')
        html_parts.append(f'<p><strong>Exported:</strong> {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}</p>')
        html_parts.append(f'<p><strong>Conversations:</strong> {len(conversations)}</p>')
        html_parts.append('</div>')
        
        for conversation, messages in zip(conversations, all_messages):
            html_parts.append('<div class="conversation">')
            
            # Conversation header
            html_parts.append(f'<h2 class="conversation-title">{self._escape_html(conversation.title)}</h2>')
            
            if conversation.description:
                html_parts.append(f'<p><strong>Description:</strong> {self._escape_html(conversation.description)}</p>')
            
            if options.include_timestamps:
                html_parts.append(f'<p><strong>Created:</strong> {conversation.created_at.strftime("%Y-%m-%d %H:%M:%S")}</p>')
                html_parts.append(f'<p><strong>Updated:</strong> {conversation.updated_at.strftime("%Y-%m-%d %H:%M:%S")}</p>')
            
            if conversation.tags:
                tags_html = ', '.join(f'<code>{self._escape_html(tag)}</code>' for tag in conversation.tags)
                html_parts.append(f'<p><strong>Tags:</strong> {tags_html}</p>')
            
            # Messages
            for message in messages:
                css_class = f"message message-{message.type.value}"
                html_parts.append(f'<div class="{css_class}">')
                
                # Message header
                header_parts = [f'<span class="message-type">{message.type.value.title()}</span>']
                if options.include_timestamps:
                    header_parts.append(f'<span class="timestamp">{message.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</span>')
                
                html_parts.append(f'<div class="message-header">{" | ".join(header_parts)}</div>')
                
                # Content
                content = message.content
                if options.sanitize_content:
                    content = self._sanitize_content(content)
                
                # Convert markdown-style code blocks to HTML
                content = self._markdown_to_html(content)
                html_parts.append(f'<div class="message-content">{content}</div>')
                
                if options.include_tokens:
                    html_parts.append(f'<div class="metadata">Tokens: {message.token_count}</div>')
                
                html_parts.append('</div>')
            
            html_parts.append('</div>')
        
        # HTML footer
        html_parts.append('</body></html>')
        
        content = "\n".join(html_parts)
        
        if output_path:
            output_path = output_path.with_suffix('.html')
            output_path.write_text(content, encoding='utf-8')
            return ExportResult(
                success=True,
                output_path=output_path,
                file_size_bytes=output_path.stat().st_size
            )
        else:
            return ExportResult(
                success=True,
                output_content=content,
                file_size_bytes=len(content.encode('utf-8'))
            )
    
    def _export_txt(
        self,
        conversations: List[ConversationThread],
        all_messages: List[List[Message]],
        options: ExportOptions,
        output_path: Optional[Path]
    ) -> ExportResult:
        """Export to plain text format."""
        
        lines = []
        
        # Header
        lines.append("ABOV3 CONVERSATION EXPORT")
        lines.append("=" * 50)
        lines.append("")
        lines.append(f"Exported: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"Conversations: {len(conversations)}")
        lines.append("")
        
        for conversation, messages in zip(conversations, all_messages):
            # Conversation header
            lines.append(f"CONVERSATION: {conversation.title}")
            lines.append("-" * 50)
            lines.append("")
            
            if conversation.description:
                lines.append(f"Description: {conversation.description}")
                lines.append("")
            
            if options.include_timestamps:
                lines.append(f"Created: {conversation.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                lines.append(f"Updated: {conversation.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
                lines.append("")
            
            if conversation.tags:
                lines.append(f"Tags: {', '.join(conversation.tags)}")
                lines.append("")
            
            # Messages
            for i, message in enumerate(messages, 1):
                lines.append(f"[{i}] {message.type.value.upper()}")
                
                if options.include_timestamps:
                    lines.append(f"Time: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                
                if options.include_tokens:
                    lines.append(f"Tokens: {message.token_count}")
                
                lines.append("")
                
                # Content
                content = message.content
                if options.sanitize_content:
                    content = self._sanitize_content(content)
                
                # Indent message content
                content_lines = content.split('\n')
                for line in content_lines:
                    lines.append(f"    {line}")
                
                lines.append("")
                lines.append("-" * 30)
                lines.append("")
            
            lines.append("=" * 50)
            lines.append("")
        
        content = "\n".join(lines)
        
        if output_path:
            output_path = output_path.with_suffix('.txt')
            output_path.write_text(content, encoding='utf-8')
            return ExportResult(
                success=True,
                output_path=output_path,
                file_size_bytes=output_path.stat().st_size
            )
        else:
            return ExportResult(
                success=True,
                output_content=content,
                file_size_bytes=len(content.encode('utf-8'))
            )
    
    def _export_pdf(
        self,
        conversations: List[ConversationThread],
        all_messages: List[List[Message]],
        options: ExportOptions,
        output_path: Optional[Path]
    ) -> ExportResult:
        """Export to PDF format."""
        
        if not REPORTLAB_AVAILABLE:
            return ExportResult(
                success=False,
                errors=["PDF export requires reportlab package: pip install reportlab"]
            )
        
        try:
            if output_path:
                pdf_path = output_path.with_suffix('.pdf')
            else:
                pdf_path = Path(tempfile.mktemp(suffix='.pdf'))
            
            doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title = Paragraph("ABOV3 Conversation Export", styles['Title'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Export info
            export_info = f"Exported: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}<br/>Conversations: {len(conversations)}"
            story.append(Paragraph(export_info, styles['Normal']))
            story.append(Spacer(1, 20))
            
            for conversation, messages in zip(conversations, all_messages):
                # Conversation title
                conv_title = Paragraph(f"Conversation: {conversation.title}", styles['Heading1'])
                story.append(conv_title)
                story.append(Spacer(1, 12))
                
                if conversation.description:
                    desc = Paragraph(f"<b>Description:</b> {conversation.description}", styles['Normal'])
                    story.append(desc)
                    story.append(Spacer(1, 6))
                
                if options.include_timestamps:
                    timestamp_info = f"<b>Created:</b> {conversation.created_at.strftime('%Y-%m-%d %H:%M:%S')}<br/><b>Updated:</b> {conversation.updated_at.strftime('%Y-%m-%d %H:%M:%S')}"
                    story.append(Paragraph(timestamp_info, styles['Normal']))
                    story.append(Spacer(1, 6))
                
                # Messages
                for i, message in enumerate(messages, 1):
                    msg_header = f"Message {i} ({message.type.value.title()})"
                    if options.include_timestamps:
                        msg_header += f" - {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                    
                    story.append(Paragraph(msg_header, styles['Heading2']))
                    story.append(Spacer(1, 6))
                    
                    # Content
                    content = message.content
                    if options.sanitize_content:
                        content = self._sanitize_content(content)
                    
                    # Simple text processing for PDF
                    content = content.replace('\n', '<br/>')
                    content = re.sub(r'```([^`]+)```', r'<font name="Courier">\1</font>', content)
                    
                    story.append(Paragraph(content, styles['Normal']))
                    story.append(Spacer(1, 12))
                
                story.append(Spacer(1, 20))
            
            doc.build(story)
            
            if output_path:
                return ExportResult(
                    success=True,
                    output_path=pdf_path,
                    file_size_bytes=pdf_path.stat().st_size
                )
            else:
                content = pdf_path.read_bytes()
                pdf_path.unlink()  # Clean up temp file
                return ExportResult(
                    success=True,
                    output_content=base64.b64encode(content).decode('utf-8'),
                    file_size_bytes=len(content),
                    metadata={"content_type": "application/pdf", "encoding": "base64"}
                )
        
        except Exception as e:
            return ExportResult(
                success=False,
                errors=[f"PDF export failed: {str(e)}"]
            )
    
    def _export_csv(
        self,
        conversations: List[ConversationThread],
        all_messages: List[List[Message]],
        options: ExportOptions,
        output_path: Optional[Path]
    ) -> ExportResult:
        """Export to CSV format."""
        
        import csv
        from io import StringIO
        
        # CSV for messages
        output = StringIO()
        writer = csv.writer(output)
        
        # Header
        headers = [
            'conversation_id', 'conversation_title', 'message_id', 'message_type',
            'content', 'timestamp'
        ]
        
        if options.include_tokens:
            headers.append('token_count')
        if options.include_metadata:
            headers.extend(['metadata', 'model_info'])
        
        writer.writerow(headers)
        
        # Data rows
        for conversation, messages in zip(conversations, all_messages):
            for message in messages:
                row = [
                    conversation.id,
                    conversation.title,
                    message.id,
                    message.type.value,
                    self._sanitize_content(message.content) if options.sanitize_content else message.content,
                    message.timestamp.isoformat() if options.include_timestamps else ''
                ]
                
                if options.include_tokens:
                    row.append(message.token_count)
                if options.include_metadata:
                    row.append(json.dumps(message.metadata))
                    row.append(json.dumps(message.model_info) if message.model_info else '')
                
                writer.writerow(row)
        
        content = output.getvalue()
        output.close()
        
        if output_path:
            output_path = output_path.with_suffix('.csv')
            output_path.write_text(content, encoding='utf-8')
            return ExportResult(
                success=True,
                output_path=output_path,
                file_size_bytes=output_path.stat().st_size
            )
        else:
            return ExportResult(
                success=True,
                output_content=content,
                file_size_bytes=len(content.encode('utf-8'))
            )
    
    def _export_xml(
        self,
        conversations: List[ConversationThread],
        all_messages: List[List[Message]],
        options: ExportOptions,
        output_path: Optional[Path]
    ) -> ExportResult:
        """Export to XML format."""
        
        xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_parts.append('<export>')
        xml_parts.append(f'  <info exported_at="{datetime.utcnow().isoformat()}" tool="ABOV3 Ollama" format="xml"/>')
        xml_parts.append('  <conversations>')
        
        for conversation, messages in zip(conversations, all_messages):
            xml_parts.append(f'    <conversation id="{conversation.id}">')
            xml_parts.append(f'      <title><![CDATA[{conversation.title}]]></title>')
            
            if conversation.description:
                xml_parts.append(f'      <description><![CDATA[{conversation.description}]]></description>')
            
            xml_parts.append(f'      <status>{conversation.status.value}</status>')
            
            if options.include_timestamps:
                xml_parts.append(f'      <created_at>{conversation.created_at.isoformat()}</created_at>')
                xml_parts.append(f'      <updated_at>{conversation.updated_at.isoformat()}</updated_at>')
            
            if conversation.tags:
                xml_parts.append('      <tags>')
                for tag in conversation.tags:
                    xml_parts.append(f'        <tag><![CDATA[{tag}]]></tag>')
                xml_parts.append('      </tags>')
            
            xml_parts.append('      <messages>')
            
            for message in messages:
                xml_parts.append(f'        <message id="{message.id}" type="{message.type.value}">')
                
                content = message.content
                if options.sanitize_content:
                    content = self._sanitize_content(content)
                
                xml_parts.append(f'          <content><![CDATA[{content}]]></content>')
                
                if options.include_timestamps:
                    xml_parts.append(f'          <timestamp>{message.timestamp.isoformat()}</timestamp>')
                
                if options.include_tokens:
                    xml_parts.append(f'          <token_count>{message.token_count}</token_count>')
                
                xml_parts.append('        </message>')
            
            xml_parts.append('      </messages>')
            xml_parts.append('    </conversation>')
        
        xml_parts.append('  </conversations>')
        xml_parts.append('</export>')
        
        content = '\n'.join(xml_parts)
        
        if output_path:
            output_path = output_path.with_suffix('.xml')
            output_path.write_text(content, encoding='utf-8')
            return ExportResult(
                success=True,
                output_path=output_path,
                file_size_bytes=output_path.stat().st_size
            )
        else:
            return ExportResult(
                success=True,
                output_content=content,
                file_size_bytes=len(content.encode('utf-8'))
            )
    
    def _export_yaml(
        self,
        conversations: List[ConversationThread],
        all_messages: List[List[Message]],
        options: ExportOptions,
        output_path: Optional[Path]
    ) -> ExportResult:
        """Export to YAML format."""
        
        try:
            import yaml
        except ImportError:
            return ExportResult(
                success=False,
                errors=["YAML export requires PyYAML package: pip install PyYAML"]
            )
        
        data = {
            'export_info': {
                'format': 'yaml',
                'version': '1.0',
                'exported_at': datetime.utcnow().isoformat(),
                'tool': 'ABOV3 Ollama'
            },
            'conversations': []
        }
        
        for conversation, messages in zip(conversations, all_messages):
            conv_data = conversation.to_dict()
            
            # Process messages
            conv_messages = []
            for message in messages:
                msg_data = message.to_dict()
                
                if options.sanitize_content:
                    msg_data["content"] = self._sanitize_content(msg_data["content"])
                
                conv_messages.append(msg_data)
            
            conv_data["messages"] = conv_messages
            data["conversations"].append(conv_data)
        
        content = yaml.dump(data, default_flow_style=False, allow_unicode=True, indent=2)
        
        if output_path:
            output_path = output_path.with_suffix('.yaml')
            output_path.write_text(content, encoding='utf-8')
            return ExportResult(
                success=True,
                output_path=output_path,
                file_size_bytes=output_path.stat().st_size
            )
        else:
            return ExportResult(
                success=True,
                output_content=content,
                file_size_bytes=len(content.encode('utf-8'))
            )
    
    def _extract_code_blocks(self, messages: List[Message], languages: List[CodeLanguage]) -> List[CodeBlock]:
        """Extract code blocks from messages."""
        
        code_blocks = []
        
        for message in messages:
            content = message.content
            
            # Extract code blocks for each language
            for language in languages:
                if language in self.code_patterns:
                    for pattern in self.code_patterns[language]:
                        matches = re.finditer(pattern, content, re.DOTALL | re.MULTILINE)
                        
                        for match in matches:
                            code_content = match.group(1).strip()
                            if code_content:
                                # Try to determine filename from content or context
                                filename = self._guess_filename(code_content, language)
                                
                                code_block = CodeBlock(
                                    language=language,
                                    content=code_content,
                                    source_message_id=message.id,
                                    line_start=content[:match.start()].count('\n') + 1,
                                    line_end=content[:match.end()].count('\n') + 1,
                                    filename=filename
                                )
                                code_blocks.append(code_block)
        
        return code_blocks
    
    def _export_code_blocks(self, code_blocks: List[CodeBlock], base_path: Optional[Path], options: ExportOptions) -> None:
        """Export extracted code blocks to separate files."""
        
        if not base_path:
            return
        
        code_dir = base_path.parent / f"{base_path.stem}_code"
        code_dir.mkdir(exist_ok=True)
        
        # Group by language
        by_language = {}
        for block in code_blocks:
            if block.language not in by_language:
                by_language[block.language] = []
            by_language[block.language].append(block)
        
        # Export each language to separate directory
        for language, blocks in by_language.items():
            lang_dir = code_dir / language.value
            lang_dir.mkdir(exist_ok=True)
            
            for i, block in enumerate(blocks, 1):
                if block.filename:
                    filename = block.filename
                else:
                    ext = self._get_file_extension(language)
                    filename = f"code_{i}{ext}"
                
                file_path = lang_dir / filename
                
                # Add header comment with source info
                header = self._generate_code_header(block, language)
                content = header + block.content
                
                file_path.write_text(content, encoding='utf-8')
        
        # Create index file
        index_lines = [
            "# Extracted Code Blocks",
            "",
            f"Extracted {len(code_blocks)} code blocks from conversations.",
            ""
        ]
        
        for language, blocks in by_language.items():
            index_lines.append(f"## {language.value.title()}")
            index_lines.append("")
            
            for block in blocks:
                filename = block.filename or f"code_{blocks.index(block) + 1}"
                index_lines.append(f"- `{filename}` (from message {block.source_message_id})")
            
            index_lines.append("")
        
        index_path = code_dir / "README.md"
        index_path.write_text("\n".join(index_lines), encoding='utf-8')
    
    def _guess_filename(self, code_content: str, language: CodeLanguage) -> Optional[str]:
        """Guess filename from code content."""
        
        # Look for filename hints in comments
        filename_patterns = [
            r'#\s*(?:file|filename):\s*([^\s\n]+)',
            r'//\s*(?:file|filename):\s*([^\s\n]+)',
            r'/\*\s*(?:file|filename):\s*([^\s\n]+)\s*\*/',
            r'<!--\s*(?:file|filename):\s*([^\s\n]+)\s*-->'
        ]
        
        for pattern in filename_patterns:
            match = re.search(pattern, code_content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Look for class/function names as filename hints
        if language == CodeLanguage.PYTHON:
            class_match = re.search(r'class\s+(\w+)', code_content)
            if class_match:
                return f"{class_match.group(1).lower()}.py"
            
            func_match = re.search(r'def\s+(\w+)', code_content)
            if func_match:
                return f"{func_match.group(1).lower()}.py"
        
        elif language == CodeLanguage.JAVA:
            class_match = re.search(r'(?:public\s+)?class\s+(\w+)', code_content)
            if class_match:
                return f"{class_match.group(1)}.java"
        
        elif language == CodeLanguage.JAVASCRIPT:
            func_match = re.search(r'function\s+(\w+)', code_content)
            if func_match:
                return f"{func_match.group(1).lower()}.js"
        
        return None
    
    def _get_file_extension(self, language: CodeLanguage) -> str:
        """Get file extension for language."""
        
        extensions = {
            CodeLanguage.PYTHON: ".py",
            CodeLanguage.JAVASCRIPT: ".js",
            CodeLanguage.TYPESCRIPT: ".ts",
            CodeLanguage.JAVA: ".java",
            CodeLanguage.CSHARP: ".cs",
            CodeLanguage.CPP: ".cpp",
            CodeLanguage.C: ".c",
            CodeLanguage.GO: ".go",
            CodeLanguage.RUST: ".rs",
            CodeLanguage.PHP: ".php",
            CodeLanguage.RUBY: ".rb",
            CodeLanguage.SHELL: ".sh",
            CodeLanguage.SQL: ".sql",
            CodeLanguage.HTML: ".html",
            CodeLanguage.CSS: ".css",
            CodeLanguage.YAML: ".yaml",
            CodeLanguage.JSON: ".json",
            CodeLanguage.XML: ".xml",
            CodeLanguage.OTHER: ".txt"
        }
        
        return extensions.get(language, ".txt")
    
    def _generate_code_header(self, block: CodeBlock, language: CodeLanguage) -> str:
        """Generate header comment for code block."""
        
        comment_styles = {
            CodeLanguage.PYTHON: "#",
            CodeLanguage.SHELL: "#",
            CodeLanguage.YAML: "#",
            CodeLanguage.RUBY: "#",
            CodeLanguage.JAVASCRIPT: "//",
            CodeLanguage.TYPESCRIPT: "//",
            CodeLanguage.JAVA: "//",
            CodeLanguage.CSHARP: "//",
            CodeLanguage.CPP: "//",
            CodeLanguage.C: "//",
            CodeLanguage.GO: "//",
            CodeLanguage.RUST: "//",
            CodeLanguage.PHP: "//",
            CodeLanguage.SQL: "--",
            CodeLanguage.HTML: "<!--",
            CodeLanguage.CSS: "/*",
            CodeLanguage.XML: "<!--"
        }
        
        comment_char = comment_styles.get(language, "#")
        
        if comment_char == "<!--":
            return f"<!-- Extracted from message {block.source_message_id} -->\n\n"
        elif comment_char == "/*":
            return f"/* Extracted from message {block.source_message_id} */\n\n"
        else:
            return f"{comment_char} Extracted from message {block.source_message_id}\n\n"
    
    def _sanitize_content(self, content: str) -> str:
        """Sanitize content for security."""
        
        # Use security manager to detect and remove malicious patterns
        is_safe, issues = self.security_manager.is_content_safe(content)
        
        if not is_safe:
            logger.warning(f"Content sanitization detected issues: {issues}")
            # Basic sanitization - remove potentially dangerous content
            content = re.sub(r'<script[^>]*>.*?</script>', '[SCRIPT REMOVED]', content, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(r'javascript:', '[JAVASCRIPT REMOVED]', content, flags=re.IGNORECASE)
            content = re.sub(r'on\w+\s*=', '[EVENT HANDLER REMOVED]', content, flags=re.IGNORECASE)
        
        return content
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        
        replacements = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;'
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return text
    
    def _markdown_to_html(self, text: str) -> str:
        """Convert basic markdown to HTML."""
        
        # Simple markdown conversion
        # Code blocks
        text = re.sub(r'```([^`]+)```', r'<pre><code>\1</code></pre>', text, flags=re.DOTALL)
        text = re.sub(r'`([^`\n]+)`', r'<code>\1</code>', text)
        
        # Bold and italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
        
        # Line breaks
        text = text.replace('\n', '<br/>')
        
        return text
    
    async def async_export_conversation(
        self,
        conversation_id: str,
        options: ExportOptions,
        output_path: Optional[Path] = None
    ) -> ExportResult:
        """Async version of export for large conversations."""
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.export_conversation, conversation_id, options, output_path)
    
    async def async_export_multiple_conversations(
        self,
        conversation_ids: List[str],
        options: ExportOptions,
        output_path: Optional[Path] = None
    ) -> ExportResult:
        """Async version of multi-export for large datasets."""
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.export_multiple_conversations, conversation_ids, options, output_path)