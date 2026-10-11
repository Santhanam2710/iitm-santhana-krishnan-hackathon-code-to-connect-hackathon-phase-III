// Pre-renders the deck icons to PNG (needs react, react-dom, react-icons, sharp). Output is committed,
// so rebuilding the deck only needs pptxgenjs.
const React = require("react");
const { renderToStaticMarkup } = require("react-dom/server");
const sharp = require("sharp");
const fi = require("react-icons/fi");
const ICONS = { news: "FiRss", resolve: "FiCrosshair", score: "FiActivity", act: "FiZap", shield: "FiShield", layers: "FiLayers",
  chart: "FiTrendingUp", bank: "FiBriefcase", alert: "FiAlertTriangle", globe: "FiGlobe", clock: "FiClock", check: "FiCheckCircle",
  cpu: "FiCpu", database: "FiDatabase", eye: "FiEye", target: "FiTarget", git: "FiGitBranch", users: "FiUsers" };
const COLORS = { amber: "#B4540B", white: "#FFFFFF", ink: "#0D1626" };
(async () => {
  for (const [name, comp] of Object.entries(ICONS)) {
    for (const [cname, col] of Object.entries(COLORS)) {
      const svg = renderToStaticMarkup(React.createElement(fi[comp], { size: 256, color: col }));
      await sharp(Buffer.from(svg)).resize(256, 256).png().toFile(`icons/${name}_${cname}.png`);
    }
  }
  console.log("icons written");
})();
