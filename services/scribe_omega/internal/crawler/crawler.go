package crawler

import (
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/adamvangrover/adam/services/scribe_omega/internal/filters"
)

type FileData struct {
	Path    string
	Content []byte
}

type RepoData struct {
	Root  string
	Files []FileData
}

func Crawl(root string, processor *filters.FileProcessor) (*RepoData, error) {
	var files []FileData

	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		// Skip hidden directories like .git
		if d.IsDir() {
			if strings.HasPrefix(d.Name(), ".") && d.Name() != "." && d.Name() != "./" {
				return filepath.SkipDir
			}
			return nil
		}

		relPath, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}

		if !processor.ShouldIncludeFile(relPath) {
			return nil
		}

		content, err := os.ReadFile(path)
		if err != nil {
			// Skip files we can't read
			return nil
		}

		processed := processor.ProcessContent(relPath, content)
		files = append(files, FileData{
			Path:    relPath,
			Content: processed,
		})

		return nil
	})

	if err != nil {
		return nil, err
	}

	return &RepoData{
		Root:  root,
		Files: files,
	}, nil
}
