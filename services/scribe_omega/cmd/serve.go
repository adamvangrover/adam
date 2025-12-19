package cmd

import (
	"embed"
	"fmt"
	"net/http"

	"github.com/spf13/cobra"
	"github.com/adamvangrover/adam/services/scribe_omega/internal/crawler"
	"github.com/adamvangrover/adam/services/scribe_omega/internal/filters"
	"github.com/adamvangrover/adam/services/scribe_omega/internal/fetcher"
	"github.com/adamvangrover/adam/services/scribe_omega/internal/transformer"
)

//go:embed web_static/*
var content embed.FS

var port string

var serveCmd = &cobra.Command{
	Use:   "serve",
	Short: "Start the Scribe-Omega web server",
	RunE: func(cmd *cobra.Command, args []string) error {
		http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/" {
				// Serve from embedded FS
				data, err := content.ReadFile("web_static/index.html")
				if err != nil {
					http.Error(w, "Could not load UI", http.StatusInternalServerError)
					return
				}
				w.Write(data)
				return
			}
		})

		http.HandleFunc("/api/process", func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
				return
			}

			tier := r.FormValue("tier")
			path := r.FormValue("path")

			if path == "" {
				w.Write([]byte(`<div id="output-content" class="text-red-500">Error: Path is required</div>`))
				return
			}

			// Handle remote URLs or local paths
			targetPath, cleanup, err := fetcher.FetchRepo(path)
			if err != nil {
				w.Write([]byte(fmt.Sprintf(`<div id="output-content" class="text-red-500">Error fetching repo: %v</div>`, err)))
				return
			}
			defer cleanup()

			processor := filters.NewFileProcessor(filters.Tier(tier))
			data, err := crawler.Crawl(targetPath, processor)
			if err != nil {
				w.Write([]byte(fmt.Sprintf(`<div id="output-content" class="text-red-500">Error processing: %v</div>`, err)))
				return
			}

			// Update root in data to be the original path/url for display purposes,
			// otherwise it shows the temp dir path.
			data.Root = path

			markdown := transformer.GenerateMarkdown(data, tier)

			// Wrap in the div expected by HTMX
			fmt.Fprintf(w, `<div id="output-content">%s</div>`, markdown)
		})

		fmt.Printf("Scribe-Omega Web Server listening on port %s\n", port)
		return http.ListenAndServe(":"+port, nil)
	},
}

func init() {
	serveCmd.Flags().StringVarP(&port, "port", "p", "8080", "Port to listen on")
	rootCmd.AddCommand(serveCmd)
}
