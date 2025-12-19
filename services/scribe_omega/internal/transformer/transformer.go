package transformer

import (
	"bytes"
	"fmt"
	"strings"
	"time"

	"github.com/adamvangrover/adam/services/scribe_omega/internal/crawler"
)

func GenerateMarkdown(data *crawler.RepoData, tier string) string {
	var sb strings.Builder

	writeHeader(&sb, data.Root, tier)
	writeManifest(&sb, data.Files)

	// File Content
	for _, f := range data.Files {
		ext := filepathExt(f.Path)
		sb.WriteString(fmt.Sprintf("--- FILE: %s ---\n", f.Path))
		sb.WriteString(fmt.Sprintf("```%s\n", ext))
		sb.Write(f.Content)
		if !bytes.HasSuffix(f.Content, []byte("\n")) {
			sb.WriteString("\n")
		}
		sb.WriteString("```\n\n")
	}

	return sb.String()
}

func GenerateSpec(data *crawler.RepoData, tier string) string {
	var sb strings.Builder
	writeHeader(&sb, data.Root, tier)
	writeManifest(&sb, data.Files)
	sb.WriteString("\n*Spec Mode: File contents omitted.*\n")
	return sb.String()
}

func writeHeader(sb *strings.Builder, root, tier string) {
	sb.WriteString(fmt.Sprintf("# Repository Export: %s\n", root))
	sb.WriteString(fmt.Sprintf("- **Generated**: %s\n", time.Now().Format(time.RFC3339)))
	sb.WriteString(fmt.Sprintf("- **Tier**: %s\n", tier))
	sb.WriteString("\n---\n\n")
}

func writeManifest(sb *strings.Builder, files []crawler.FileData) {
	sb.WriteString("## Manifest\n\n")
	sb.WriteString("```text\n")
	for _, f := range files {
		sb.WriteString(f.Path + "\n")
	}
	sb.WriteString("```\n\n")
	sb.WriteString("---\n\n")
}

func filepathExt(path string) string {
	parts := strings.Split(path, ".")
	if len(parts) > 1 {
		return parts[len(parts)-1]
	}
	return ""
}
