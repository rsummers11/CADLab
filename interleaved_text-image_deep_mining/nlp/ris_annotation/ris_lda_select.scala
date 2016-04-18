import scalanlp.io._;
import scalanlp.stage._;
import scalanlp.stage.text._;
import scalanlp.text.tokenize._;
import scalanlp.pipes.Pipes.global._;

import edu.stanford.nlp.tmt.stage._;
import edu.stanford.nlp.tmt.model.lda._;
import edu.stanford.nlp.tmt.model.llda._;

import java.io._;

object ris_lda_select{
  def main(args: Array[String]) = {
    val source = CSVFile("../data/ris_reports.csv") ~> IDColumn(1);

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
      TermMinimumDocumentCountFilter(3) ~>   // filter terms in <4 docs
      TermStopListFilter(List("0001","0002","Report")) ~>
      TermDynamicStopListFilter(10) ~>       // filter out 30 most common terms
      DocumentMinimumLengthFilter(5)         // take only docs with >=5 terms
    }

    // set aside 80 percent of the input text as training data ...
    val numTrain = text.data.size * 4 / 5;

    // build a training dataset
    val training = LDADataset(text ~> Take(numTrain));
     
    // build a test dataset, using term index from the training dataset 
    val testing  = LDADataset(text ~> Drop(numTrain));

    // a list of pairs of (number of topics, perplexity)
    var scores = List.empty[(Int,Double)];


    // loop over various numbers of topics, training and evaluating each model
    for (numTopics <- List(10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,510,520,530,540,550,560,570,580,590,600,710,720,730,740,750,760,770,780,790,700,810,820,830,840,850,860,870,880,890,800,910,920,930,940,950,960,970,980,990,1000)) {
      val params = LDAModelParams(numTopics = numTopics, dataset = training);
      val output = file("lda-"+training.signature+"-"+params.signature);
      val model = TrainCVB0LDA(params, training, output=null, maxIterations=500);//5000);
      
      println("[perplexity] computing at "+numTopics);

      val perplexity = model.computePerplexity(testing);
      
      println("[perplexity] perplexity at "+numTopics+" topics: "+perplexity);

      scores :+= (numTopics, perplexity);
    }

    val writer = new PrintWriter(new File("risLDASel_tr" + args(0) + ".txt" ))
    for ((numTopics,perplexity) <- scores) {
      writer.write("[perplexity] perplexity at "+numTopics+" topics: "+perplexity);
    }
    writer.close()
  }
}
