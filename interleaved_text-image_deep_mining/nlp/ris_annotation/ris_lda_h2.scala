import scalanlp.io._;
import scalanlp.stage._;
import scalanlp.stage.text._;
import scalanlp.text.tokenize._;
import scalanlp.pipes.Pipes.global._;

import edu.stanford.nlp.tmt.stage._;
import edu.stanford.nlp.tmt.model.lda._;
import edu.stanford.nlp.tmt.model.llda._;

import java.io._;

object ris_lda{
  def main(args: Array[String]) = {

    for (h2tpi <- (args(2).toInt-1)*10 to ((args(2).toInt-1)*10+9)){

      val source = CSVFile("../data/ris_reports_per_topic_h2/ris_reports_per_h1_topic_" + 
        h2tpi + ".csv") ~> IDColumn(1);

      val tokenizer = {
        SimpleEnglishTokenizer() ~>            // tokenize on space and punctuation
        CaseFolder() ~>                        // lowercase everything
        WordsAndNumbersOnlyFilter() ~>         // ignore non-words and non-numbers
        MinimumLengthFilter(3)                 // take terms with >=3 characters
      }

      val text = {
        source ~>                              // read from the source file
        Column(3) ~>                           // select column containing text
        TokenizeWith(tokenizer) ~>             // tokenize with tokenizer above
        TermCounter() ~>                       // collect counts (needed below)
        TermMinimumDocumentCountFilter(3) ~>   // filter terms in <3 docs
        TermStopListFilter(List("0001","0002","Report")) ~>
        TermDynamicStopListFilter(10) ~>       // filter out 30 most common terms
        DocumentMinimumLengthFilter(5)         // take only docs with >=5 terms
      }

      // display information about the loaded dataset
      println("Description of the loaded text field:");
      println(text.description);

      println();
      println("------------------------------------");
      println();

      println("Terms in the stop list:");
      for (term <- text.meta[TermStopList]) {
        println("  " + term);
      }

      val dataset = LDADataset(text);

      // define the model parameters
      val params = LDAModelParams(numTopics = args(0).toInt, dataset = dataset,
        topicSmoothing = 0.01, termSmoothing = 0.01);

      // Name of the output model folder to generate
      val modelPath = file("../data/ris_reports_per_topic_h2/ris_lda_h2_h1topic_" + 
        h2tpi + "-" +
        "ntp" + args(0) + "-" + "ntr" + args(1) + "-" + 
        dataset.signature+"-"+params.signature);

      // Trains the model: the model (and intermediate models) are written to the
      // output folder.  If a partially trained model with the same dataset and
      // parameters exists in that folder, training will be resumed.
      TrainCVB0LDA(params, dataset, output=modelPath, maxIterations=10000);

      // To use the Gibbs sampler for inference, instead use
      // TrainGibbsLDA(params, dataset, output=modelPath, maxIterations=1500);

    }

  }
}
