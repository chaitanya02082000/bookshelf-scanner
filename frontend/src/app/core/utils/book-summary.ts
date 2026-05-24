function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatInlineMarkdown(value: string): string {
  return escapeHtml(value)
    .replace(/\[([^\]]+)\]\[\d+\]/g, "$1")
    .replace(/\*\*\*([^*]+)\*\*\*/g, "<strong><em>$1</em></strong>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

export function formatBookSummaryHtml(summary?: string | null): string {
  if (!summary?.trim()) {
    return "";
  }

  const normalized = summary
    .replace(/\s+(#{1,6}\s+)/g, "\n$1")
    .replace(/\s+-\s+/g, "\n- ")
    .replace(/\s+Followed by:/g, "\nFollowed by:")
    .replace(/\[(\d+)\]:\s+https?:\/\/\S+/g, "")
    .trim();

  const lines = normalized.split(/\n+/).map((line) => line.trim()).filter(Boolean);
  const blocks: string[] = [];
  let paragraphLines: string[] = [];
  let listItems: string[] = [];

  const flushParagraph = () => {
    if (!paragraphLines.length) {
      return;
    }
    blocks.push(`<p>${formatInlineMarkdown(paragraphLines.join(" "))}</p>`);
    paragraphLines = [];
  };

  const flushList = () => {
    if (!listItems.length) {
      return;
    }
    blocks.push(`<ul>${listItems.map((item) => `<li>${formatInlineMarkdown(item)}</li>`).join("")}</ul>`);
    listItems = [];
  };

  for (const line of lines) {
    if (/^#{1,6}\s+/.test(line)) {
      flushParagraph();
      flushList();
      const title = line.replace(/^#{1,6}\s+/, "");
      blocks.push(`<h4>${formatInlineMarkdown(title)}</h4>`);
      continue;
    }

    if (/^-\s+/.test(line)) {
      flushParagraph();
      listItems.push(line.replace(/^-\s+/, ""));
      continue;
    }

    flushList();
    paragraphLines.push(line);
  }

  flushParagraph();
  flushList();
  return blocks.join("");
}
