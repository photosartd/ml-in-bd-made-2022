ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.10"

lazy val root = (project in file("."))
  .settings(
    name := "Scala Linear Regression"
  )

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % "2.1.0"
)