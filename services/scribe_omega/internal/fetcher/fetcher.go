package fetcher

import (
	"archive/zip"
	"bytes"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// FetchRepo handles local paths and remote GitHub URLs.
// If it's a URL, it downloads the zip, extracts it to a temp dir, and returns the path.
// It returns the path to crawl and a cleanup function.
func FetchRepo(input string) (string, func(), error) {
	if strings.HasPrefix(input, "http://") || strings.HasPrefix(input, "https://") {
		return downloadZip(input)
	}
	return input, func() {}, nil
}

func downloadZip(url string) (string, func(), error) {
	// Convert GitHub URL to zip download URL if necessary
	// Example: https://github.com/user/repo -> https://github.com/user/repo/archive/refs/heads/main.zip
	// This is a naive conversion; better to use API or expect direct zip, but let's try to be smart.

	downloadURL := url
	if strings.Contains(url, "github.com") && !strings.HasSuffix(url, ".zip") {
		// Assuming main branch for now if not specified.
		// A more robust solution would check for main/master or use the API.
		// Let's try appending /archive/refs/heads/main.zip
		// But we don't know the default branch.
		// Safer approach: /archive/HEAD.zip usually works for the default branch.
		downloadURL = strings.TrimSuffix(url, "/") + "/archive/HEAD.zip"
	}

	resp, err := http.Get(downloadURL)
	if err != nil {
		return "", nil, fmt.Errorf("failed to download repo: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", nil, fmt.Errorf("failed to download repo: status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", nil, fmt.Errorf("failed to read response body: %w", err)
	}

	zipReader, err := zip.NewReader(bytes.NewReader(body), int64(len(body)))
	if err != nil {
		return "", nil, fmt.Errorf("failed to open zip: %w", err)
	}

	// Create temp dir
	tmpDir, err := os.MkdirTemp("", "scribe-omega-")
	if err != nil {
		return "", nil, fmt.Errorf("failed to create temp dir: %w", err)
	}

	cleanup := func() {
		os.RemoveAll(tmpDir)
	}

	// Extract
	for _, f := range zipReader.File {
		fpath := filepath.Join(tmpDir, f.Name)

		// Check for Zip Slip
		if !strings.HasPrefix(fpath, filepath.Clean(tmpDir)+string(os.PathSeparator)) {
			cleanup()
			return "", nil, fmt.Errorf("illegal file path: %s", fpath)
		}

		if f.FileInfo().IsDir() {
			os.MkdirAll(fpath, os.ModePerm)
			continue
		}

		if err := os.MkdirAll(filepath.Dir(fpath), os.ModePerm); err != nil {
			cleanup()
			return "", nil, err
		}

		dstFile, err := os.OpenFile(fpath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
		if err != nil {
			cleanup()
			return "", nil, err
		}

		srcFile, err := f.Open()
		if err != nil {
			dstFile.Close()
			cleanup()
			return "", nil, err
		}

		_, err = io.Copy(dstFile, srcFile)
		srcFile.Close()
		dstFile.Close()
		if err != nil {
			cleanup()
			return "", nil, err
		}
	}

	// GitHub zip files usually contain a top-level directory (e.g. repo-main/).
	// We want to return that inner directory if it exists and is the only one.
	entries, err := os.ReadDir(tmpDir)
	if err == nil && len(entries) == 1 && entries[0].IsDir() {
		return filepath.Join(tmpDir, entries[0].Name()), cleanup, nil
	}

	return tmpDir, cleanup, nil
}
