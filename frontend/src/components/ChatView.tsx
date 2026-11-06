/**
 * ChatView Component — RepoPilot LangGraph Conversational Agent
 *
 * Full-featured conversational interface with:
 * - Durable session threads persisted in PostgreSQL / Checkpointer
 * - Modular architecture (Sidebar, Thread, InputBar)
 * - Real-time SSE token streaming
 * - Active tool invocation status badges
 * - Markdown & syntax-formatted code rendering
 * - Keyboard shortcuts (Enter to send, Shift+Enter for newline)
 */

import { type FormEvent, useEffect, useRef, useState } from "react";
import type { ActiveToolExecution, ChatMessageItem, ChatSession, Repository } from "../types";
import {
  createChatSession,
  deleteChatSession,
  getChatHistory,
  listChatSessions,
  streamChatMessage,
} from "../services/api";
import ChatInputBar from "./chat/ChatInputBar";
import ChatSidebar from "./chat/ChatSidebar";
import ChatThread from "./chat/ChatThread";

/**
 * Filter out internal execution logs, tool dumps, or empty thinking states.
 */
function filterCleanMessages(rawMessages: ChatMessageItem[]): ChatMessageItem[] {
  if (!Array.isArray(rawMessages)) return [];

  return rawMessages.filter((msg) => {
    if (!msg || typeof msg !== "object") return false;
    if (msg.role === "tool") return false;
    if (msg.role !== "user" && msg.role !== "assistant") return false;

    const content = typeof msg.content === "string" ? msg.content.trim() : "";
    if (!content) return false;

    const lower = content.toLowerCase();
    if (
      lower === "thinking" ||
      lower === "*thinking*" ||
      lower === "thinking…" ||
      lower === "thinking..."
    ) {
      return false;
    }

    if (lower.startsWith("tool execution output received")) {
      return false;
    }

    return true;
  });
}

interface ChatViewProps {
  repositories: Repository[];
  initialRepoId?: number | string | null;
  onRepoSelect?: (repoId: string) => void;
}

export default function ChatView({
  repositories,
  initialRepoId,
  onRepoSelect,
}: ChatViewProps) {
  const [selectedRepoId, setSelectedRepoId] = useState<string>(
    initialRepoId ? String(initialRepoId) : ""
  );

  // Chat sessions state
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>("");
  const [loadingSessions, setLoadingSessions] = useState<boolean>(false);

  // Conversation & streaming state
  const [messages, setMessages] = useState<ChatMessageItem[]>([]);
  const [question, setQuestion] = useState<string>("");
  const [asking, setAsking] = useState<boolean>(false);
  const [streamingMessage, setStreamingMessage] = useState<string>("");
  const [activeTools, setActiveTools] = useState<ActiveToolExecution[]>([]);
  const [error, setError] = useState<string>("");

  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const accumulatedStreamRef = useRef<string>("");
  const chatThreadRef = useRef<HTMLDivElement>(null);

  // Sync external repo switch
  useEffect(() => {
    if (initialRepoId && String(initialRepoId) !== selectedRepoId) {
      // oxlint-disable-next-line react/set-state-in-effect
      setSelectedRepoId(String(initialRepoId));
      setSessions([]);
      setCurrentSessionId("");
      setMessages([]);
      setError("");
    }
  }, [initialRepoId, selectedRepoId]);

  const selectedRepo = repositories.find((r) => r.id === Number(selectedRepoId));

  // Auto-scroll inside chat thread container
  useEffect(() => {
    if (chatThreadRef.current) {
      if (streamingMessage) {
        chatThreadRef.current.scrollTop = chatThreadRef.current.scrollHeight;
      } else {
        chatThreadRef.current.scrollTo({
          top: chatThreadRef.current.scrollHeight,
          behavior: "smooth",
        });
      }
    }
  }, [messages, asking, streamingMessage, activeTools]);

  // Load chat sessions when selected repository changes
  useEffect(() => {
    if (!selectedRepoId) return;

    let isMounted = true;
    // oxlint-disable-next-line react/set-state-in-effect
    setLoadingSessions(true);

    listChatSessions(selectedRepoId)
      .then(async (sessList) => {
        if (!isMounted) return;
        if (sessList && sessList.length > 0) {
          setSessions(sessList);
          const firstSessionId = sessList[0].id;
          setCurrentSessionId(firstSessionId);
          try {
            const history = await getChatHistory(firstSessionId);
            if (isMounted) {
              setMessages(filterCleanMessages(history.messages || []));
            }
          } catch {
            if (isMounted) setMessages([]);
          }
        } else {
          // Auto-create initial session
          try {
            const newSession = await createChatSession(selectedRepoId, "New Conversation");
            if (isMounted) {
              setSessions([newSession]);
              setCurrentSessionId(newSession.id);
              setMessages([]);
            }
          } catch {
            if (isMounted) setSessions([]);
          }
        }
      })
      .catch((err: Error) => {
        if (isMounted) setError(err.message || "Failed to load chat sessions.");
      })
      .finally(() => {
        if (isMounted) setLoadingSessions(false);
      });

    return () => {
      isMounted = false;
    };
  }, [selectedRepoId]);

  /** Switch active session thread. */
  async function handleSelectSession(sessionId: string) {
    if (sessionId === currentSessionId || asking) return;
    setCurrentSessionId(sessionId);
    setError("");
    setMessages([]);

    try {
      const history = await getChatHistory(sessionId);
      setMessages(filterCleanMessages(history.messages || []));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to load session history.");
    }
  }

  /** Create a new conversation session. */
  async function handleCreateSession() {
    if (!selectedRepoId || asking) return;
    setError("");

    try {
      const newSession = await createChatSession(selectedRepoId, "New Conversation");
      setSessions((prev) => [newSession, ...prev]);
      setCurrentSessionId(newSession.id);
      setMessages([]);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to create new conversation.");
    }
  }

  /** Delete a conversation session. */
  async function handleDeleteSession(event: React.MouseEvent, sessionId: string) {
    event.stopPropagation();
    if (asking) return;

    try {
      await deleteChatSession(sessionId);
      const remaining = sessions.filter((s) => s.id !== sessionId);
      setSessions(remaining);

      if (sessionId === currentSessionId) {
        if (remaining.length > 0) {
          await handleSelectSession(remaining[0].id);
        } else {
          await handleCreateSession();
        }
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to delete chat session.");
    }
  }

  /** Handle repository selection change. */
  function handleRepoChange(nextRepoId: string) {
    setSelectedRepoId(nextRepoId);
    setSessions([]);
    setCurrentSessionId("");
    setMessages([]);
    setError("");
    if (onRepoSelect) onRepoSelect(nextRepoId);
  }

  /** Inject an example question into textarea. */
  function applyExampleQuestion(q: string) {
    setQuestion(q);
    textareaRef.current?.focus();
  }

  /** Submit message and stream response via SSE. */
  async function handleAsk(event: FormEvent) {
    event.preventDefault();

    const trimmed = question.trim();
    if (!trimmed || !selectedRepoId || asking) return;

    let activeSessionId = currentSessionId;
    if (!activeSessionId) {
      try {
        const newSession = await createChatSession(selectedRepoId, "New Conversation");
        setSessions((prev) => [newSession, ...prev]);
        setCurrentSessionId(newSession.id);
        activeSessionId = newSession.id;
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : "Failed to create conversation session.");
        return;
      }
    }

    setQuestion("");
    setError("");
    setAsking(true);
    setStreamingMessage("");
    setActiveTools([]);
    accumulatedStreamRef.current = "";

    // Append user message to thread
    setMessages((prev) => [...prev, { role: "user", content: trimmed }]);

    await streamChatMessage(selectedRepoId, activeSessionId, trimmed, {
      onToken: (delta) => {
        accumulatedStreamRef.current += delta;
        setStreamingMessage(accumulatedStreamRef.current);
      },
      onToolStart: (toolName) => {
        setActiveTools((prev) => [
          ...prev.filter((t) => t.tool !== toolName),
          { tool: toolName, status: "running" },
        ]);
      },
      onToolEnd: (toolName) => {
        setActiveTools((prev) =>
          prev.map((t) => (t.tool === toolName ? { ...t, status: "completed" } : t))
        );
      },
      onError: (err) => {
        setError(err.message || "Generation error encountered.");
        setAsking(false);
      },
      onDone: () => {
        const finalContent = accumulatedStreamRef.current;
        if (finalContent) {
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: finalContent },
          ]);
        }
        setStreamingMessage("");
        setActiveTools([]);
        setAsking(false);

        // Refresh session list to show auto-generated title
        listChatSessions(selectedRepoId)
          .then(setSessions)
          .catch(() => {});
      },
    });
  }

  return (
    <div className="chat-layout">
      {/* Sidebar */}
      <ChatSidebar
        repositories={repositories}
        selectedRepoId={selectedRepoId}
        onRepoChange={handleRepoChange}
        sessions={sessions}
        currentSessionId={currentSessionId}
        loadingSessions={loadingSessions}
        asking={asking}
        onSelectSession={handleSelectSession}
        onCreateSession={handleCreateSession}
        onDeleteSession={handleDeleteSession}
        onSelectExample={applyExampleQuestion}
      />

      {/* Main Chat Area */}
      <div className="chat-main">
        <ChatThread
          selectedRepoId={selectedRepoId}
          selectedRepoFullName={selectedRepo?.full_name}
          messages={messages}
          asking={asking}
          streamingMessage={streamingMessage}
          activeTools={activeTools}
          error={error}
          chatThreadRef={chatThreadRef}
          bottomRef={bottomRef}
        />

        <ChatInputBar
          question={question}
          onChangeQuestion={setQuestion}
          onSubmit={handleAsk}
          selectedRepoId={selectedRepoId}
          selectedRepoName={selectedRepo?.full_name}
          asking={asking}
          textareaRef={textareaRef}
        />
      </div>
    </div>
  );
}
