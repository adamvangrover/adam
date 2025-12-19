package cmd

import (
	"fmt"
	"os"

	"github.com/atotto/clipboard"
	"github.com/spf13/cobra"
	"github.com/adamvangrover/adam/services/scribe_omega/internal/crawler"
	"github.com/adamvangrover/adam/services/scribe_omega/internal/filters"
	"github.com/adamvangrover/adam/services/scribe_omega/internal/fetcher"
	"github.com/adamvangrover/adam/services/scribe_omega/internal/transformer"
)

var (
	targetDir string
	tier      string
	output    string
	specOnly  bool
	noClipboard bool
)

var rootCmd = &cobra.Command{
	Use:   "scribe-omega",
	Short: "Scribe-Omega: Repository to Markdown Converter",
	Long:  `A high-performance tool to convert code repositories into LLM-optimized Markdown.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		// Default to current directory if not specified
		if targetDir == "" {
			wd, err := os.Getwd()
			if err != nil {
				return err
			}
			targetDir = wd
		}

		fmt.Printf("Processing %s at tier %s...\n", targetDir, tier)

		// Handle remote URLs or local paths
		path, cleanup, err := fetcher.FetchRepo(targetDir)
		if err != nil {
			return err
		}
		defer cleanup()

		processor := filters.NewFileProcessor(filters.Tier(tier))
		data, err := crawler.Crawl(path, processor)
		if err != nil {
			return err
		}

		// Set root back to original input for display
		data.Root = targetDir

		var markdown string
		if specOnly {
			markdown = transformer.GenerateSpec(data, tier)
		} else {
			markdown = transformer.GenerateMarkdown(data, tier)
		}

		if output != "" {
			err := os.WriteFile(output, []byte(markdown), 0644)
			if err != nil {
				return err
			}
			fmt.Printf("Output written to %s\n", output)
		} else {
			fmt.Println(markdown)
		}

		if !noClipboard {
			if err := clipboard.WriteAll(markdown); err == nil {
				fmt.Println("\n[Copied to clipboard]")
			}
		}

		return nil
	},
}

func Execute() error {
	return rootCmd.Execute()
}

func init() {
	rootCmd.Flags().StringVarP(&targetDir, "dir", "d", "", "Directory or URL to scan")
	rootCmd.Flags().StringVarP(&tier, "tier", "t", "standard", "Tier: micro (1k), standard (10k), full (100k)")
	rootCmd.Flags().StringVarP(&output, "output", "o", "", "Output file path (default: stdout)")
	rootCmd.Flags().BoolVar(&specOnly, "spec-only", false, "Generate only the architecture spec/manifest")
	rootCmd.Flags().BoolVar(&noClipboard, "no-clipboard", false, "Disable clipboard copying")
}
