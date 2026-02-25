package ai.router

/**
 * A candidate provider+model pair in a routing rule.
 * When [model] is blank, the router picks the first available model from that provider.
 */
data class RouteTarget(val providerName: String, val model: String = "")
