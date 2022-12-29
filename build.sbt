import Dependencies._

ThisBuild / scalaVersion     := "2.13.8"
ThisBuild / version          := "0.1.0-SNAPSHOT"
ThisBuild / organization     := "com.example"
ThisBuild / organizationName := "example"

lazy val root = (project in file("."))
  .settings(
    name := "tensorflow-scala-exercice",
    libraryDependencies += scalaTest % Test,
    libraryDependencies ++= Seq(
      "org.platanios" %% "tensorflow" % "0.6.5" classifier "linux",
    ),
  )

// See https://www.scala-sbt.org/1.x/docs/Using-Sonatype.html for instructions on how to publish to Sonatype.
