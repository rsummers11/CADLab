import scalanlp.io._;
import scalanlp.stage._;
import scalanlp.stage.text._;
import scalanlp.text.tokenize._;
import scalanlp.pipes.Pipes.global._;

import edu.stanford.nlp.tmt.stage._;
import edu.stanford.nlp.tmt.model.lda._;
import edu.stanford.nlp.tmt.model.llda._;

import java.io._;

object ris_lda_infer{
  def main(args: Array[String]) = {

    val modelFList = List("ris_lda_h2_h1topic_0-ntp10-ntr1-4226202c-10-fb6db87a","ris_lda_h2_h1topic_1-ntp10-ntr1-d293c84b-10-a0ec7c48","ris_lda_h2_h1topic_2-ntp10-ntr1-6301706a-10-813a5c49","ris_lda_h2_h1topic_3-ntp10-ntr1-f36f1889-10-4d8d3c9e","ris_lda_h2_h1topic_4-ntp10-ntr1-83dcc0a8-10-f6c1b7dc","ris_lda_h2_h1topic_5-ntp10-ntr1-144a68c7-10-630d245c","ris_lda_h2_h1topic_6-ntp10-ntr1-a4b810e6-10-70cffba4","ris_lda_h2_h1topic_7-ntp10-ntr1-3525b905-10-5c65e0ea","ris_lda_h2_h1topic_8-ntp10-ntr1-c5936124-10-66127da4","ris_lda_h2_h1topic_9-ntp10-ntr1-56010943-10-0f8ce3e8","ris_lda_h2_h1topic_10-ntp10-ntr1-2ed6c837-10-20aeb43c","ris_lda_h2_h1topic_11-ntp10-ntr1-bf447056-10-1409d936","ris_lda_h2_h1topic_12-ntp10-ntr1-4fb21875-10-82f2adff","ris_lda_h2_h1topic_13-ntp10-ntr1-e01fc094-10-1898f0a9","ris_lda_h2_h1topic_14-ntp10-ntr1-708d68b3-10-e56cc499","ris_lda_h2_h1topic_15-ntp10-ntr1-00fb10d2-10-99cd4898","ris_lda_h2_h1topic_16-ntp10-ntr1-9168b8f1-10-146b48fb","ris_lda_h2_h1topic_17-ntp10-ntr1-21d66110-10-a78c0841","ris_lda_h2_h1topic_18-ntp10-ntr1-b244092f-10-0dbcf402","ris_lda_h2_h1topic_19-ntp10-ntr1-42b1b14e-10-89371d40","ris_lda_h2_h1topic_20-ntp10-ntr1-ac1e23f8-10-3371afdb","ris_lda_h2_h1topic_21-ntp10-ntr1-3c8bcc17-10-cc020dc3","ris_lda_h2_h1topic_22-ntp10-ntr1-ccf97436-10-afdb8885","ris_lda_h2_h1topic_23-ntp10-ntr1-5d671c55-10-3d0bfc43","ris_lda_h2_h1topic_24-ntp10-ntr1-edd4c474-10-673079e6","ris_lda_h2_h1topic_25-ntp10-ntr1-7e426c93-10-c46820fd","ris_lda_h2_h1topic_26-ntp10-ntr1-0eb014b2-10-9b1a502a","ris_lda_h2_h1topic_27-ntp10-ntr1-9f1dbcd1-10-b7b8ad1e","ris_lda_h2_h1topic_28-ntp10-ntr1-2f8b64f0-10-65b5257e","ris_lda_h2_h1topic_29-ntp10-ntr1-bff90d0f-10-da8f6386","ris_lda_h2_h1topic_30-ntp10-ntr1-29657fb9-10-5b1daf71","ris_lda_h2_h1topic_31-ntp10-ntr1-b9d327d8-10-7aa2e706","ris_lda_h2_h1topic_32-ntp10-ntr1-4a40cff7-10-cdc85177","ris_lda_h2_h1topic_33-ntp10-ntr1-daae7816-10-c1bf0224","ris_lda_h2_h1topic_34-ntp10-ntr1-6b1c2035-10-32a319c1","ris_lda_h2_h1topic_35-ntp10-ntr1-fb89c854-10-65141dc0","ris_lda_h2_h1topic_36-ntp10-ntr1-8bf77073-10-072c1881","ris_lda_h2_h1topic_37-ntp10-ntr1-1c651892-10-d7d2be1c","ris_lda_h2_h1topic_38-ntp10-ntr1-acd2c0b1-10-3dc888c0","ris_lda_h2_h1topic_39-ntp10-ntr1-3d4068d0-10-b604b9bc","ris_lda_h2_h1topic_40-ntp10-ntr1-a6acdb7a-10-d2467d04","ris_lda_h2_h1topic_41-ntp10-ntr1-371a8399-10-caa23e1a","ris_lda_h2_h1topic_42-ntp10-ntr1-c7882bb8-10-84031780","ris_lda_h2_h1topic_43-ntp10-ntr1-57f5d3d7-10-9adc9462","ris_lda_h2_h1topic_44-ntp10-ntr1-e8637bf6-10-26da20e1","ris_lda_h2_h1topic_45-ntp10-ntr1-78d12415-10-b824b5bc","ris_lda_h2_h1topic_46-ntp10-ntr1-093ecc34-10-8449493b","ris_lda_h2_h1topic_47-ntp10-ntr1-99ac7453-10-021a907f","ris_lda_h2_h1topic_48-ntp10-ntr1-2a1a1c72-10-8787417d","ris_lda_h2_h1topic_49-ntp10-ntr1-ba87c491-10-878329de","ris_lda_h2_h1topic_50-ntp10-ntr1-23f4373b-10-a95bf827","ris_lda_h2_h1topic_51-ntp10-ntr1-b461df5a-10-28d668e4","ris_lda_h2_h1topic_52-ntp10-ntr1-44cf8779-10-b5be8801","ris_lda_h2_h1topic_53-ntp10-ntr1-d53d2f98-10-e2538583","ris_lda_h2_h1topic_54-ntp10-ntr1-65aad7b7-10-39ce0401","ris_lda_h2_h1topic_55-ntp10-ntr1-f6187fd6-10-203e93ff","ris_lda_h2_h1topic_56-ntp10-ntr1-868627f5-10-0ada3100","ris_lda_h2_h1topic_57-ntp10-ntr1-16f3d014-10-38dcf029","ris_lda_h2_h1topic_58-ntp10-ntr1-a7617833-10-fae0f5c7","ris_lda_h2_h1topic_59-ntp10-ntr1-37cf2052-10-79459980","ris_lda_h2_h1topic_60-ntp10-ntr1-a13b92fc-10-24c4bdba","ris_lda_h2_h1topic_61-ntp10-ntr1-31a93b1b-10-33e6ece6","ris_lda_h2_h1topic_62-ntp10-ntr1-c216e33a-10-66ac8c63","ris_lda_h2_h1topic_63-ntp10-ntr1-52848b59-10-cf282120","ris_lda_h2_h1topic_64-ntp10-ntr1-e2f23378-10-3a3a0c9f","ris_lda_h2_h1topic_65-ntp10-ntr1-735fdb97-10-3948f8c7","ris_lda_h2_h1topic_66-ntp10-ntr1-03cd83b6-10-98f5375c","ris_lda_h2_h1topic_67-ntp10-ntr1-943b2bd5-10-a01aec46","ris_lda_h2_h1topic_68-ntp10-ntr1-24a8d3f4-10-1ac38582","ris_lda_h2_h1topic_69-ntp10-ntr1-b5167c13-10-c99be008","ris_lda_h2_h1topic_70-ntp10-ntr1-1e82eebd-10-72c60922","ris_lda_h2_h1topic_71-ntp10-ntr1-aef096dc-10-870870c8","ris_lda_h2_h1topic_72-ntp10-ntr1-3f5e3efb-10-d12880c2","ris_lda_h2_h1topic_73-ntp10-ntr1-cfcbe71a-10-dc548983","ris_lda_h2_h1topic_74-ntp10-ntr1-60398f39-10-db95c198","ris_lda_h2_h1topic_75-ntp10-ntr1-f0a73758-10-56ae345c","ris_lda_h2_h1topic_76-ntp10-ntr1-8114df77-10-d04205c3","ris_lda_h2_h1topic_77-ntp10-ntr1-11828796-10-6e7243dc","ris_lda_h2_h1topic_78-ntp10-ntr1-a1f02fb5-10-d41a0cc4","ris_lda_h2_h1topic_79-ntp10-ntr1-325dd7d4-10-2de4510c");
    for (h2tpi <- (args(2).toInt-1)*10 to ((args(2).toInt-1)*10+9)){

      val modelPath = file("../data/ris_reports_per_topic_h2/" + modelFList(h2tpi));

      println("Loading "+modelPath);
      val model = LoadCVB0LDA(modelPath);

      val source = CSVFile("../data/ris_reports_per_topic_h2/ris_reports_per_h1_topic_" + 
        h2tpi + ".csv") ~> IDColumn(1);

      val text = {
        source ~>                              // read from the source file
        Column(3) ~>                           // select column containing text
        TokenizeWith(model.tokenizer.get)      // tokenize with existing model's tokenizer
      }

      // Base name of output files to generate
      val output = file(modelPath, source.meta[java.io.File].getName.replaceAll(".csv",""));

      // turn the text into a dataset ready to be used with LDA
      val dataset = LDADataset(text, termIndex = model.termIndex);

      println("Writing document distributions to "+output+"-document-topic-distributions.csv");
      val perDocTopicDistributions = InferCVB0DocumentTopicDistributions(model, dataset);
      CSVFile(output+"-document-topic-distributuions.csv").write(perDocTopicDistributions);

      println("Writing topic usage to "+output+"-usage.csv");
      val usage = QueryTopicUsage(model, dataset, perDocTopicDistributions);
      CSVFile(output+"-usage.csv").write(usage);

      println("Estimating per-doc per-word topic distributions");
      val perDocWordTopicDistributions = EstimatePerWordTopicDistributions(
        model, dataset, perDocTopicDistributions);

      println("Writing top terms to "+output+"-top-terms.csv");
      val topTerms = QueryTopTerms(model, dataset, perDocWordTopicDistributions, numTopTerms=50);
      CSVFile(output+"-top-terms.csv").write(topTerms);

    }

  }
}
