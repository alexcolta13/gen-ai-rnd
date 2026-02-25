import cli.InteractiveCLI
import cli.SingleShotCLI

fun main(args: Array<String>) {
    if (args.isEmpty()) {
        InteractiveCLI().run()
    } else {
        SingleShotCLI().run(args)
    }
}
