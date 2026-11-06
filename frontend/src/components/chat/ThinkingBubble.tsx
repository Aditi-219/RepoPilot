/**
 * ThinkingBubble Component
 *
 * Animated indicator shown while the backend is planning tool calls or generating tokens.
 */

export default function ThinkingBubble() {
  return (
    <div className="chat-bubble chat-bubble--assistant chat-bubble--thinking">
      <div className="chat-thinking" aria-label="Analyzing question and preparing tools">
        <span className="chat-thinking-dot" aria-hidden="true" />
        <span className="chat-thinking-dot" aria-hidden="true" />
        <span className="chat-thinking-dot" aria-hidden="true" />
        <span className="chat-thinking-label">
          Analyzing question and preparing tools
        </span>
      </div>
    </div>
  );
}
