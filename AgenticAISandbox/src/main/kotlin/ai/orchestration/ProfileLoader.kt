package ai.orchestration

import com.google.gson.Gson
import java.io.File

object ProfileLoader {
    private val gson = Gson()
    private val profilesDir = File("profiles")

    /**
     * Loads and deserializes a profile from profiles/<name>.json.
     * Returns null if the file does not exist or cannot be parsed.
     */
    fun load(name: String): ProfileConfig? {
        val file = File(profilesDir, "$name.json")
        if (!file.exists()) return null
        return runCatching {
            gson.fromJson(file.readText(), ProfileConfig::class.java)
        }.getOrElse {
            System.err.println("[ProfileLoader] Failed to parse '$name': ${it.message}")
            null
        }
    }

    /**
     * Lists all profiles available in the profiles/ directory.
     * Returns a list of (filename-without-extension, description) pairs.
     */
    fun listAvailable(): List<Pair<String, String>> {
        if (!profilesDir.isDirectory) return emptyList()
        return profilesDir
            .listFiles { f -> f.isFile && f.extension == "json" }
            ?.sortedBy { it.name }
            ?.mapNotNull { file ->
                runCatching {
                    val config = gson.fromJson(file.readText(), ProfileConfig::class.java)
                    file.nameWithoutExtension to (config.description ?: "")
                }.getOrNull()
            }
            ?: emptyList()
    }
}
