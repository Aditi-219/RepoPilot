import type { RefObject } from "react";
import type { ActiveToolExecution, ChatMessageItem } from "../../types";
import AssistantBubble from "./AssistantBubble";
import ThinkingBubble from "./ThinkingBubble";
import ToolExecutionBadge from "./ToolExecutionBadge";

function MessageIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}

interface ChatThreadProps {
  selectedRepoId: string;
  selectedRepoFullName?: string;
  messages: ChatMessageItem[];
  asking: boolean;
  streamingMessage: string;
  activeTools: ActiveToolExecution[];
  error?: string;
  chatThreadRef: RefObject<HTMLDivElement | null>;
  bottomRef: RefObject<HTMLDivElement | null>;
}

export default function ChatThread({
  selectedRepoId,
  selectedRepoFullName,
  messages,
  asking,
  streamingMessage,
  activeTools,
  error,
  chatThreadRef,
  bottomRef,
}: ChatThreadProps) {
  return (
    <div
      ref={chatThreadRef}
      className="chat-thread"
      role="log"
      aria-live="polite"
      aria-label="Conversation"
    >
      {/* Empty state — no repo selected */}
      {!selectedRepoId && (
        <div className="chat-empty">
          <div className="chat-empty-icon-wrap">
            <MessageIcon />
          </div>
          <h3 className="chat-empty-heading">Select a repository to start</h3>
          <p className="chat-empty-subheading">
            Choose an indexed repository from the sidebar, then ask any
            question about its codebase.
          </p>
        </div>
      )}

      {/* Empty state — repo selected but no messages yet */}
      {selectedRepoId && messages.length === 0 && !asking && (
        <div className="chat-empty">
          <div className="chat-empty-icon-wrap">
            <SearchIcon />
          </div>
          <h3 className="chat-empty-heading">
            Ask about{" "}
            <strong style={{ color: "var(--color-text-primary)" }}>
              {selectedRepoFullName}
            </strong>
          </h3>
          <p className="chat-empty-subheading">
            A LangGraph agent with access to repository metadata, file trees,
            and source files will assist you with streaming responses and code citations.
          </p>
        </div>
      )}

      {/* Message list */}
      {messages.map((msg, idx) => {
        if (msg.role === "user") {
          return (
            <div key={msg.id || idx} className="chat-message chat-message--user">
              <div className="chat-bubble chat-bubble--user">
                <div className="chat-bubble-text">{msg.content}</div>
              </div>
            </div>
          );
        }

        if (msg.role === "assistant" && msg.content) {
          return (
            <div key={msg.id || idx} className="chat-message chat-message--assistant">
              <AssistantBubble
                id={msg.id || `msg-${idx}`}
                content={msg.content}
                citations={msg.citations}
                model={msg.model}
              />
            </div>
          );
        }

        return null;
      })}

      {/* Active Tool Executions */}
      {activeTools.length > 0 && (
        <div className="chat-tools-container" aria-live="polite">
          {activeTools.map((t, idx) => (
            <ToolExecutionBadge key={idx} tool={t.tool} status={t.status} />
          ))}
        </div>
      )}

      {/* Streaming Assistant Bubble */}
      {asking && streamingMessage && (
        <div className="chat-message chat-message--assistant">
          <AssistantBubble
            id="streaming-bubble"
            content={streamingMessage}
            isStreaming={true}
          />
        </div>
      )}

      {/* Animated typing indicator when waiting before first token */}
      {asking && !streamingMessage && activeTools.length === 0 && (
        <div className="chat-message chat-message--assistant">
          <ThinkingBubble />
        </div>
      )}

      {/* Error alert */}
      {error && (
        <div
          className="rp-alert rp-alert-error chat-error-alert"
          role="alert"
          aria-live="assertive"
        >
          {error}
        </div>
      )}

      {/* Scroll anchor */}
      <div ref={bottomRef} />
    </div>
  );
}
