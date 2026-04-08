import { complete, type Model, type Api } from "@mariozechner/pi-ai";
import type { ExtensionAPI, ExtensionCommandContext } from "@mariozechner/pi-coding-agent";

const skillPattern = /^\/skill(?:\:| +)(\S+)(?: +([\s\S]*))?$/;

const SUMMARY_PROMPT =
  "Summarize the user's request in 5-10 words max. Output ONLY the summary, nothing else. No quotes, no punctuation at the end.";
const SESSION_SUMMARY_PROMPT =
  "Summarize this conversation as a short session title in 5-10 words max. Focus on the main task, decision, or outcome. Output ONLY the title, nothing else. No quotes, no punctuation at the end.";

const HAIKU_MODEL_ID = "claude-haiku-4-5";
const MAX_CONVERSATION_CHARS = 12_000;

type SessionBranchEntry = {
  type: string;
  summary?: string;
  message?: {
    role?: string;
    content?: unknown;
  };
};

type TextContentBlock = {
  type?: string;
  text?: string;
};

type SummaryContentBlock = {
  type: string;
  text?: string;
  thinking?: string;
  redacted?: boolean;
};

type SummarizeContext = Pick<ExtensionCommandContext, "sessionManager" | "hasUI" | "ui" | "model" | "modelRegistry">;
type ModelPickResult =
  | { ok: true; model: Model<Api>; apiKey?: string; headers?: Record<string, string> }
  | { ok: false; reason: string };
type SessionSummaryResult =
  | { ok: true; summary: string }
  | { ok: false; reason: string };

type ResolvedAuth = {
  apiKey?: string;
  headers?: Record<string, string>;
};

function hasRequestAuth(auth: ResolvedAuth): boolean {
  return !!auth.apiKey || !!(auth.headers && Object.keys(auth.headers).length > 0);
}

async function pickCheapModel(ctx: {
  model: Model<Api> | null;
  modelRegistry: {
    find: (p: string, id: string) => Model<Api> | undefined;
    getApiKeyAndHeaders: (m: Model<Api>) => Promise<{ ok: true; apiKey?: string; headers?: Record<string, string> } | { ok: false; error: string }>;
  };
}): Promise<ModelPickResult> {
  let haikuFailure = `anthropic/${HAIKU_MODEL_ID} is not available`;
  const haiku = ctx.modelRegistry.find("anthropic", HAIKU_MODEL_ID);
  if (haiku) {
    const auth = await ctx.modelRegistry.getApiKeyAndHeaders(haiku);
    if ("error" in auth) {
      haikuFailure = `anthropic/${HAIKU_MODEL_ID} auth failed: ${auth.error}`;
    } else if (hasRequestAuth(auth)) {
      return { ok: true, model: haiku, apiKey: auth.apiKey, headers: auth.headers };
    } else {
      haikuFailure = `anthropic/${HAIKU_MODEL_ID} has no API key or auth headers`;
    }
  }

  if (ctx.model) {
    const auth = await ctx.modelRegistry.getApiKeyAndHeaders(ctx.model);
    if ("error" in auth) {
      return { ok: false, reason: `${haikuFailure}. Current model ${ctx.model.provider}/${ctx.model.id} auth failed: ${auth.error}` };
    }
    if (!hasRequestAuth(auth)) {
      return { ok: false, reason: `${haikuFailure}. Current model ${ctx.model.provider}/${ctx.model.id} has no API key or auth headers` };
    }
    return { ok: true, model: ctx.model, apiKey: auth.apiKey, headers: auth.headers };
  }

  return { ok: false, reason: `${haikuFailure}. No current model is selected for fallback` };
}

function extractText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";

  return content
    .filter((block): block is TextContentBlock => !!block && typeof block === "object")
    .filter((block) => block.type === "text" && typeof block.text === "string")
    .map((block) => block.text)
    .join("\n")
    .trim();
}

function trimConversation(text: string): string {
  if (text.length <= MAX_CONVERSATION_CHARS) return text;

  const separator = "\n\n[Earlier conversation omitted for brevity]\n\n";
  const headLength = Math.floor((MAX_CONVERSATION_CHARS - separator.length) * 0.4);
  const tailLength = MAX_CONVERSATION_CHARS - separator.length - headLength;
  return `${text.slice(0, headLength).trimEnd()}${separator}${text.slice(-tailLength).trimStart()}`;
}

function buildConversationText(entries: SessionBranchEntry[]): string {
  const sections: string[] = [];

  for (const entry of entries) {
    if (entry.type === "compaction" && typeof entry.summary === "string" && entry.summary.trim()) {
      sections.push(`Earlier summary: ${entry.summary.trim()}`);
      continue;
    }

    if (entry.type === "branch_summary" && typeof entry.summary === "string" && entry.summary.trim()) {
      sections.push(`Branch summary: ${entry.summary.trim()}`);
      continue;
    }

    if (entry.type !== "message") continue;

    const role = entry.message?.role;
    if (role !== "user" && role !== "assistant") continue;

    const text = extractText(entry.message?.content);
    if (!text) continue;

    sections.push(`${role === "user" ? "User" : "Assistant"}: ${text}`);
  }

  return trimConversation(sections.join("\n\n").trim());
}

function extractSummary(response: { content: SummaryContentBlock[] }): string {
  return response.content
    .filter((c): c is SummaryContentBlock & { type: "text"; text: string } => c.type === "text" && typeof c.text === "string")
    .map((c) => c.text)
    .join("")
    .replace(/\s+/g, " ")
    .trim();
}

function describeResponseContent(response: { content: SummaryContentBlock[] }): string {
  const types = response.content.map((block) => {
    if (block.type === "thinking") {
      if (block.redacted) return "thinking:redacted";
      const hasThinking = typeof block.thinking === "string" && block.thinking.trim().length > 0;
      return hasThinking ? "thinking" : "thinking:empty";
    }
    if (block.type === "text") {
      const hasText = typeof block.text === "string" && block.text.trim().length > 0;
      return hasText ? "text" : "text:empty";
    }
    return block.type;
  });

  return types.length > 0 ? types.join(", ") : "none";
}

function preserveSkillPrefix(currentName: string | undefined, summary: string): string {
  const prefix = currentName?.match(/^(\[[^\]]+\])(?:\s|$)/)?.[1];
  return prefix ? `${prefix} ${summary}` : summary;
}

async function summarizeSession(ctx: SummarizeContext): Promise<SessionSummaryResult> {
  const conversationText = buildConversationText(ctx.sessionManager.getBranch() as SessionBranchEntry[]);
  if (!conversationText) return { ok: false, reason: "No conversation text found" };

  const cheap = await pickCheapModel(ctx);
  if ("reason" in cheap) return { ok: false, reason: cheap.reason };

  const messages = [
    {
      role: "user" as const,
      content: [{ type: "text" as const, text: `<conversation>\n${conversationText}\n</conversation>` }],
      timestamp: Date.now(),
    },
  ];

  const response = await complete(
    cheap.model,
    {
      systemPrompt: SESSION_SUMMARY_PROMPT,
      messages,
    },
    { apiKey: cheap.apiKey, headers: cheap.headers },
  );

  const summary = extractSummary(response);
  if (summary) return { ok: true, summary };

  const retryResponse = await complete(
    cheap.model,
    {
      systemPrompt: `${SESSION_SUMMARY_PROMPT} Reply with plain text only in a single short line.`,
      messages,
    },
    {
      apiKey: cheap.apiKey,
      headers: cheap.headers,
      reasoning: "minimal",
      thinking: { enabled: false },
    },
  );

  const retrySummary = extractSummary(retryResponse);
  if (retrySummary) return { ok: true, summary: retrySummary };

  return {
    ok: false,
    reason: `Model returned no usable title text (initial blocks: ${describeResponseContent(response)}; retry blocks: ${describeResponseContent(retryResponse)})`,
  };
}

export default function (pi: ExtensionAPI) {
  let named = false;

  const setSessionName = (name: string) => {
    pi.setSessionName(name);
    named = true;
  };

  const summarizeAndRenameSession = async (ctx: SummarizeContext) => {
    const conversationText = buildConversationText(ctx.sessionManager.getBranch() as SessionBranchEntry[]);
    if (!conversationText) {
      if (ctx.hasUI) ctx.ui.notify("No conversation text found", "warning");
      return;
    }

    if (ctx.hasUI) ctx.ui.notify("Summarizing session...", "info");

    try {
      const result = await summarizeSession(ctx);
      if ("reason" in result) {
        if (ctx.hasUI) ctx.ui.notify(result.reason, "warning");
        return;
      }

      const nextName = preserveSkillPrefix(pi.getSessionName(), result.summary);
      setSessionName(nextName);
      if (ctx.hasUI) ctx.ui.notify(`Session renamed: ${nextName}`, "info");
    } catch (error) {
      if (ctx.hasUI) ctx.ui.notify(error instanceof Error ? error.message : "Failed to summarize session", "warning");
    }
  };

  pi.on("session_start", () => {
    named = !!pi.getSessionName();
  });

  pi.on("input", async (event, ctx) => {
    if (named || pi.getSessionName()) {
      named = true;
      return;
    }

    const match = event.text.match(skillPattern);
    if (!match) return;

    const skillName = match[1];
    const userPrompt = (match[2] ?? "").trim();
    named = true;

    if (!userPrompt) {
      setSessionName(`[${skillName}]`);
      return;
    }

    // Set a temporary name immediately so something shows up
    setSessionName(`[${skillName}] ${userPrompt.slice(0, 60)}`);

    // Summarize in the background with a cheap model
    const cheap = await pickCheapModel(ctx);
    if ("reason" in cheap) return;

    try {
      const response = await complete(
        cheap.model,
        {
          systemPrompt: SUMMARY_PROMPT,
          messages: [{ role: "user", content: [{ type: "text", text: userPrompt }], timestamp: Date.now() }],
        },
        { apiKey: cheap.apiKey, headers: cheap.headers },
      );

      const summary = extractSummary(response);

      if (summary) {
        setSessionName(`[${skillName}] ${summary}`);
      }
    } catch {
      // Keep the truncated name, no big deal
    }
  });

  pi.registerCommand("summarize-session", {
    description: "Summarize the current conversation and rename the session",
    handler: async (_args, ctx) => summarizeAndRenameSession(ctx),
  });

  pi.registerShortcut("ctrl+shift+r", {
    description: "Summarize the current conversation and rename the session",
    handler: async (ctx) => summarizeAndRenameSession(ctx),
  });
}
