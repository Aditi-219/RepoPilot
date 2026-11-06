import { type FormEvent, type KeyboardEvent, type RefObject } from "react";

function SendIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

interface ChatInputBarProps {
  question: string;
  onChangeQuestion: (val: string) => void;
  onSubmit: (e: FormEvent) => void;
  selectedRepoId: string;
  selectedRepoName?: string;
  asking: boolean;
  textareaRef: RefObject<HTMLTextAreaElement | null>;
}

export default function ChatInputBar({
  question,
  onChangeQuestion,
  onSubmit,
  selectedRepoId,
  selectedRepoName,
  asking,
  textareaRef,
}: ChatInputBarProps) {
  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSubmit(event);
    }
  };

  return (
    <div className="chat-input-area">
      <form onSubmit={onSubmit} className="chat-input-form" aria-label="Send a message">
        <textarea
          ref={textareaRef}
          className="chat-textarea"
          placeholder={
            selectedRepoId
              ? `Ask about ${selectedRepoName ?? "the codebase"}…`
              : "Select a repository first"
          }
          value={question}
          onChange={(e) => onChangeQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={!selectedRepoId || asking}
          rows={1}
          aria-label="Question input"
        />

        <button
          type="submit"
          className="chat-send-btn"
          disabled={!selectedRepoId || !question.trim() || asking}
          aria-label="Send question"
        >
          {asking ? (
            <span className="rp-spinner" role="status" aria-label="Sending" />
          ) : (
            <SendIcon />
          )}
        </button>
      </form>

      <p className="chat-hint">
        Enter to send &middot; Shift+Enter for new line &middot; Powered by{" "}
        <strong>LangGraph</strong> + <strong>pgvector</strong>
      </p>
    </div>
  );
}
