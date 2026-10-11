import { fireEvent, render, screen } from "@testing-library/react";
import { MouthMap } from "@/components/MouthMap";
import { RiskBadge } from "@/components/clinical";

describe("MouthMap", () => {
  it("selects a site by click and keyboard", () => {
    const onChange = vi.fn();
    render(<MouthMap value="Lip" onChange={onChange} />);
    expect(screen.getByRole("radio", { name: "Lip" })).toHaveAttribute("aria-checked", "true");
    fireEvent.click(screen.getByRole("radio", { name: "Hard palate" }));
    fireEvent.keyDown(screen.getByRole("radio", { name: "Floor of mouth" }), { key: "Enter" });
    expect(onChange).toHaveBeenNthCalledWith(1, "Hard palate");
    expect(onChange).toHaveBeenNthCalledWith(2, "Floor of mouth");
  });

  it("is a static image when used as a heat map", () => {
    render(<MouthMap counts={{ Lip: 3 }} />);
    expect(screen.getByRole("img", { name: /site map/i })).toBeInTheDocument();
    expect(screen.queryByRole("radio")).toBeNull();
  });
});

describe("RiskBadge", () => {
  it("labels each tier", () => {
    render(<><RiskBadge tier="high" /><RiskBadge tier="low" size="sm" /></>);
    expect(screen.getByText("High risk")).toBeInTheDocument();
    expect(screen.getByText("Low")).toBeInTheDocument();
  });
});
