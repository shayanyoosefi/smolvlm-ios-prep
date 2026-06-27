import Foundation

public struct SmolVLMArtifactManifest: Decodable {
    public let schemaVersion: Int
    public let generatedAt: String
    public let packageName: String
    public let modelID: String
    public let revision: String
    public let target: String
    public let runtime: Runtime?
    public let prompting: Prompting?
    public let files: [ArtifactFile]
    public let missingRequired: [String]
    public let missingOptional: [String]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case generatedAt = "generated_at"
        case packageName = "package_name"
        case modelID = "model_id"
        case revision
        case target
        case runtime
        case prompting
        case files
        case missingRequired = "missing_required"
        case missingOptional = "missing_optional"
    }

    public struct Runtime: Decodable {
        public let engine: String?
        public let iosMinimum: String?
        public let notes: [String]?

        enum CodingKeys: String, CodingKey {
            case engine
            case iosMinimum = "ios_minimum"
            case notes
        }
    }

    public struct Prompting: Decodable {
        public let chatTemplateSource: String?
        public let imageToken: String?
        public let notes: [String]?

        enum CodingKeys: String, CodingKey {
            case chatTemplateSource = "chat_template_source"
            case imageToken = "image_token"
            case notes
        }
    }

    public struct ArtifactFile: Decodable {
        public let role: String
        public let source: String
        public let path: String
        public let required: Bool
        public let present: Bool
        public let bytes: Int64?
        public let sha256: String?
    }

    public func file(role: String) throws -> ArtifactFile {
        guard let file = files.first(where: { $0.role == role && $0.present }) else {
            throw SmolVLMPackageError.missingRole(role)
        }
        return file
    }
}

public struct SmolVLMPackage {
    public let rootURL: URL
    public let manifest: SmolVLMArtifactManifest

    public static func load(from rootURL: URL) throws -> SmolVLMPackage {
        let manifestURL = rootURL.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw SmolVLMPackageError.missingManifest(manifestURL)
        }

        let data = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(SmolVLMArtifactManifest.self, from: data)
        return SmolVLMPackage(rootURL: rootURL, manifest: manifest)
    }

    public func url(forRole role: String) throws -> URL {
        let file = try manifest.file(role: role)
        return rootURL.appendingPathComponent(file.path)
    }

    public var visionEncoderURL: URL {
        get throws { try url(forRole: "vision_encoder") }
    }

    public var tokenEmbeddingURL: URL {
        get throws { try url(forRole: "token_embedding") }
    }

    public var decoderURL: URL {
        get throws { try url(forRole: "decoder") }
    }

    public var tokenizerURL: URL {
        get throws { try url(forRole: "tokenizer") }
    }

    public var tokenizerConfigURL: URL {
        get throws { try url(forRole: "tokenizer_config") }
    }

    public var imageProcessorURL: URL {
        get throws { try url(forRole: "image_processor") }
    }
}

public enum SmolVLMPackageError: Error, LocalizedError {
    case missingManifest(URL)
    case missingRole(String)

    public var errorDescription: String? {
        switch self {
        case .missingManifest(let url):
            return "SmolVLM manifest.json was not found at \(url.path)."
        case .missingRole(let role):
            return "SmolVLM artifact role is missing or not present: \(role)."
        }
    }
}

