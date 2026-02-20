class ProvenanceSystem {
    constructor() {
        this.ontology = null;
        this.lineage = [];
        this.templates = {};
        this.ready = false;
        this.subscribers = [];
    }

    async init() {
        try {
            console.log("ProvenanceSystem: Loading data...");
            const [onto, lin] = await Promise.all([
                fetch('data/provenance/ontology.json').then(r => r.json()),
                fetch('data/provenance/lineage.json').then(r => r.json())
            ]);
            this.ontology = onto;
            this.lineage = lin;
            this.ready = true;
            console.log(`ProvenanceSystem initialized with ${this.lineage.length} artifacts.`);
            this.notifyReady();
            return true;
        } catch (e) {
            console.error("ProvenanceSystem init failed:", e);
            return false;
        }
    }

    onReady(callback) {
        if (this.ready) {
            callback();
        } else {
            this.subscribers.push(callback);
        }
    }

    notifyReady() {
        this.subscribers.forEach(cb => cb());
        this.subscribers = [];
    }

    getArtifactLineage(artifactId) {
        return this.lineage.find(l => l.artifactId === artifactId);
    }

    getAllArtifacts() {
        return this.lineage.map(l => {
            const artifactNode = l.nodes.find(n => n.type === 'Artifact');
            return {
                id: l.artifactId,
                name: artifactNode ? artifactNode.label : l.artifactId,
                type: 'Artifact'
            };
        });
    }

    async loadTemplate(type) {
        if (!this.templates[type]) {
            try {
                const res = await fetch(`data/provenance/templates/${type}.json`);
                if (!res.ok) throw new Error(`Template ${type} not found`);
                this.templates[type] = await res.json();
            } catch (e) {
                console.error(e);
                return null;
            }
        }
        return this.templates[type];
    }
}

window.provenanceSystem = new ProvenanceSystem();
// Auto-init on load
document.addEventListener('DOMContentLoaded', () => {
    window.provenanceSystem.init();
});
