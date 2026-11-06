import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error("Uncaught error caught by ErrorBoundary:", error, errorInfo);
  }

  public handleReset = (): void => {
    this.setState({ hasError: false, error: null });
  };

  public render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="error-boundary-container" style={{ padding: "40px 20px", textAlign: "center" }}>
          <div className="rp-card" style={{ maxWidth: "600px", margin: "0 auto", padding: "32px" }}>
            <h2 style={{ fontSize: "20px", fontWeight: 700, marginBottom: "12px", color: "var(--color-text-primary)" }}>
              Something went wrong
            </h2>
            <p style={{ fontSize: "14px", color: "var(--color-text-secondary)", marginBottom: "20px" }}>
              {this.state.error?.message || "An unexpected error occurred in the user interface."}
            </p>
            <button
              type="button"
              className="rp-btn rp-btn-primary"
              onClick={this.handleReset}
              style={{ margin: "0 auto" }}
            >
              Try Again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
