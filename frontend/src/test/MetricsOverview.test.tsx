import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import MetricsOverview from "../components/Dashboard/MetricsOverview";

describe("MetricsOverview", () => {
  it("displays metric counts accurately", () => {
    render(<MetricsOverview totalCount={12} activeCount={3} completedCount={8} />);

    expect(screen.getByText("12")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByText("8")).toBeInTheDocument();
    expect(screen.getByText("Ready")).toBeInTheDocument();
  });

  it("does not display Ready badge when completedCount is 0", () => {
    render(<MetricsOverview totalCount={2} activeCount={2} completedCount={0} />);

    expect(screen.queryByText("Ready")).not.toBeInTheDocument();
  });
});
