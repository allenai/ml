# ML Components

Repository for low-level production-grade ML inference. The current motivating example is CRF inference. It's currently 100% Java, but can also have Scala too.

## Getting Started

This project is currently Java 8 built with `gradle`. To install `gradle` simply install via `brew install gradle` via [Homebrew](http://brew.sh). Then if you can do:
```bash
> gradle test # Run unit tests
> gradle idea # Generate IntelliJ project
```

## Project Conventions

### Documentation

You can use Markdown in your Javadoc using [Pegdown](https://github.com/sirthias/pegdown).

### Test Coverage

You can run `gradle jacoco` and this will produce a testing report

### Lombok

This project uses [Lombok](https://projectlombok.org) which requires you to enable annotation processing inside of an IDE.
[Here](https://plugins.jetbrains.com/plugin/6317) is the IntelliJ plugin and you'll need to enable annotation processing (instructions [here](https://www.jetbrains.com/idea/help/configuring-annotation-processing.html)).

Lombok has a lot of useful annotations that give you some of the nice things in Scala:

* `val` is equivalent to `final` and the right-hand-side class. It gives you type-inference via some tricks
* Checkout [`@Data`](https://projectlombok.org/features/Data.html)

### Efficient Primitive Collections

Using [GSCollections](https://github.com/goldmansachs/gs-collections) which has been found as efficient as the best libraries across a wide-range of tasks (in particular way faster than [trove](http://trove.starlight-systems.com)).

