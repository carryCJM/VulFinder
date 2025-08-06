import scala.io.Source
import scala.sys.exit

@main def execMultiple(filepaths: String): Unit = {
  // 读取文件列表
  val fileList = Source.fromFile(filepaths).getLines().toList

  for (filename <- fileList) {
    importCode.c(filename)

    val saveprefix = filename
    run.ossdataflow

    cpg.graph.E.map(node => 
      List(node.inNode.id, node.outNode.id, node.label, node.propertiesMap.get("VARIABLE"))
    ).toJson |> s"${saveprefix}_edges.json"

    cpg.graph.V.map(node => node).toJson |> s"${saveprefix}_nodes.json"

    delete
  }
}

