package filters

import (
	"bufio"
	"bytes"
	"strings"
)

type Tier string

const (
	TierMicro    Tier = "micro"
	TierStandard Tier = "standard"
	TierFull     Tier = "full"
)

// FileProcessor defines how a file should be processed based on the tier
type FileProcessor struct {
	Tier Tier
}

func NewFileProcessor(tier Tier) *FileProcessor {
	return &FileProcessor{Tier: tier}
}

// ShouldIncludeFile determines if a file should be included in the output
func (p *FileProcessor) ShouldIncludeFile(path string) bool {
	base := strings.ToLower(path)

	// Always ignore git and binary assets (basic check)
	if strings.Contains(base, ".git/") || strings.Contains(base, ".ds_store") || strings.Contains(base, "node_modules") {
		return false
	}

	// Extension checks
	isCode := strings.HasSuffix(base, ".go") || strings.HasSuffix(base, ".py") || strings.HasSuffix(base, ".js") || strings.HasSuffix(base, ".ts") || strings.HasSuffix(base, ".java")
	isConfig := strings.HasSuffix(base, "go.mod") || strings.HasSuffix(base, "package.json") || strings.HasSuffix(base, "requirements.txt")
	isReadme := strings.HasSuffix(base, "readme.md")

	switch p.Tier {
	case TierMicro:
		// Micro: README, go.mod/package.json
		return isReadme || isConfig || isCode // We include code files but will strip them to skeletons later
	case TierStandard:
		// Standard: All code files. Exclude tests and vendor.
		if strings.Contains(base, "_test.go") || strings.Contains(base, "test_") || strings.Contains(base, "vendor/") {
			return false
		}
		return isCode || isReadme || isConfig
	case TierFull:
		// Full: Everything except binary/git (already filtered)
		return true
	}
	return false
}

// ProcessContent transforms the file content based on the tier
func (p *FileProcessor) ProcessContent(path string, content []byte) []byte {
	switch p.Tier {
	case TierMicro:
		return p.extractSkeleton(path, content)
	case TierStandard:
		return p.stripCommentsAndWhitespace(path, content)
	case TierFull:
		return content
	}
	return content
}

// extractSkeleton attempts to keep only function signatures
func (p *FileProcessor) extractSkeleton(path string, content []byte) []byte {
	// Simple regex-based approach for POC.
	// This is not a full AST parser but handles common patterns.
	lines := bytes.Split(content, []byte("\n"))
	var result bytes.Buffer

	// Regex for Python/Go/JS function definitions
	// Python: def func(...): | class Class(...):
	// Go: func Func(...) ... { | type Type struct {
	// JS: function func(...) { | const func = (...) => {

	// Very naive implementation: keep lines that look like definitions, skip others.
	// For README/Configs, keep as is.
	lowerPath := strings.ToLower(path)
	if strings.HasSuffix(lowerPath, ".md") || strings.HasSuffix(lowerPath, ".json") || strings.HasSuffix(lowerPath, ".mod") {
		return content
	}

	for _, line := range lines {
		trimmed := bytes.TrimSpace(line)
		sLine := string(trimmed)
		if strings.HasPrefix(sLine, "func ") ||
		   strings.HasPrefix(sLine, "def ") ||
		   strings.HasPrefix(sLine, "class ") ||
		   strings.HasPrefix(sLine, "type ") ||
		   strings.Contains(sLine, "function ") {
			result.Write(line)
			result.WriteString("\n    ...\n")
		}
	}

	if result.Len() == 0 {
		return []byte("// [Skeleton: No significant structure found]\n")
	}
	return result.Bytes()
}

// stripCommentsAndWhitespace removes comments and empty lines
func (p *FileProcessor) stripCommentsAndWhitespace(path string, content []byte) []byte {
	scanner := bufio.NewScanner(bytes.NewReader(content))
	var result bytes.Buffer

	// Simple state for block comments
	inBlockComment := false

	for scanner.Scan() {
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)

		if trimmed == "" {
			continue
		}

		// Handle Python
		if strings.HasSuffix(path, ".py") {
			if strings.HasPrefix(trimmed, "#") {
				continue
			}
			// Triple quotes (naive)
			if strings.Contains(trimmed, `"""`) || strings.Contains(trimmed, `'''`) {
				count := strings.Count(trimmed, `"""`) + strings.Count(trimmed, `'''`)
				if count%2 != 0 {
					inBlockComment = !inBlockComment
				}
				// If the line is *just* quotes (docstring), skip it if we are entering or exiting
				// But extracting parts is hard. For standard tier, let's just strip lines that are pure comments.
			}
			if inBlockComment {
				continue
			}
		}

		// Handle Go/JS/C-like
		if strings.HasSuffix(path, ".go") || strings.HasSuffix(path, ".js") || strings.HasSuffix(path, ".ts") {
			if strings.HasPrefix(trimmed, "//") {
				continue
			}
			if strings.HasPrefix(trimmed, "/*") {
				inBlockComment = true
			}
			if inBlockComment {
				if strings.Contains(trimmed, "*/") {
					inBlockComment = false
				}
				continue
			}
		}

		result.WriteString(line + "\n")
	}
	return result.Bytes()
}
