package recommender;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import recommender.common.CommonRecomm_NoBias;
import recommender.common.RTuple;

public class NLF extends CommonRecomm_NoBias {
    public NLF() throws NumberFormatException, IOException {
		super();                                                                                                                        
		this.ifBi = false;
		this.ifBu = false;
		this.ifMu = false;
	}

	public void train() throws IOException {

		FileWriter fw = new FileWriter(
				new File("E:/code/NLF/Out/"+ this.getClass().getName().trim() + "_" + System.currentTimeMillis() + "chla_online_RMSE.txt"));
		for (int round = 1; round <= trainingRound; round++) {
			resetNMFAuxArrays(); 
			for (RTuple tempRating : trainData) {
				double ratingHat = this.getLocPrediction(tempRating.iUserID, tempRating.iItemID, this.ifMu, this.ifBi,
						this.ifBu);
				userBiasUp[tempRating.iUserID] += tempRating.dRating;
				userBiasDown[tempRating.iUserID] += ratingHat;
				itemBiasUp[tempRating.iItemID] += tempRating.dRating;
				itemBiasDown[tempRating.iItemID] += ratingHat;

				for (int dimen = 0; dimen < featureDimension; dimen++) {
					userFactorUp[tempRating.iUserID][dimen] += itemFeatureArray[tempRating.iItemID][dimen]
							* tempRating.dRating;

					userFactorDown[tempRating.iUserID][dimen] += itemFeatureArray[tempRating.iItemID][dimen]
							* ratingHat;

					itemFactorUp[tempRating.iItemID][dimen] += userFeatureArray[tempRating.iUserID][dimen]
							* tempRating.dRating;

					itemFactorDown[tempRating.iItemID][dimen] += userFeatureArray[tempRating.iUserID][dimen]
							* ratingHat;
				}
			}
			for (int tempUserID = 1; tempUserID <= maxUserID; tempUserID++) {
				userBiasDown[tempUserID] += userBias[tempUserID] * userRSetSize[tempUserID] * lambda;
				if (userBiasDown[tempUserID] != 0)
					userBias[tempUserID] *= (userBiasUp[tempUserID] / userBiasDown[tempUserID]);

				for (int dimen = 0; dimen < featureDimension; dimen++) {
					userFactorDown[tempUserID][dimen] += userFeatureArray[tempUserID][dimen] * userRSetSize[tempUserID]
							* lambda;
					if (userFactorDown[tempUserID][dimen] != 0)
						userFeatureArray[tempUserID][dimen] *= (userFactorUp[tempUserID][dimen]
								/ userFactorDown[tempUserID][dimen]);
				}
			}

			for (int tempItemID = 1; tempItemID <= maxItemID; tempItemID++) {
				itemBiasDown[tempItemID] += itemBias[tempItemID] * itemRSetSize[tempItemID] * lambda;
				if (itemBiasDown[tempItemID] != 0)
					itemBias[tempItemID] *= (itemBiasUp[tempItemID] / itemBiasDown[tempItemID]);

				for (int dimen = 0; dimen < featureDimension; dimen++) {
					itemFactorDown[tempItemID][dimen] += itemFeatureArray[tempItemID][dimen] * itemRSetSize[tempItemID]
							* lambda;
					if (itemFactorDown[tempItemID][dimen] != 0) {
						itemFeatureArray[tempItemID][dimen] *= (itemFactorUp[tempItemID][dimen]
								/ itemFactorDown[tempItemID][dimen]);
					}
				}
			}
			double sumRMSE = 0, sumCount = 0,sumMAE=0,value=0;
			for (RTuple tempTestRating : testData) {
				double actualRating = tempTestRating.dRating;
				double ratinghat = this.getLocPrediction(tempTestRating.iUserID, tempTestRating.iItemID);           
				sumRMSE += Math.pow((actualRating - ratinghat), 2);
				sumMAE+=Math.abs(actualRating-ratinghat);
				sumCount++;
			}		
			double RMSE = Math.sqrt(sumRMSE / sumCount);
			double MAE = sumMAE / sumCount ;
			fw.write(RMSE + "\n");
			fw.flush();	
			value=RMSE;
			System.out.println( round+" RMSE:"+ value);
			this.cacheMinFeatures();
		    if (this.minMAE > value) {
		    	if(this.minMAE - value <= 1e-6){
		    		this.minMAE = value;
					this.minRound = round;
		    		System.out.println("Program early stopping!,minRMSE="+this.minMAE);
		    		break;
		    		}
		    	this.minMAE = value;
				this.minRound = round;
				} else {
					
						if ((round - this.minRound) == 1) {
							this.cacheMinFeatures();
						}
						if ((round - this.minRound) > delayCount ) {
							break;					
							}
							}
		    System.out.println( "minRMSE="+this.minMAE);
		    
		}
		fw.close();	
		this.outputMinFeatures();
	}

	
	public static void main(String[] argv) throws NumberFormatException, IOException {
		long startTime = System.currentTimeMillis();    
		CommonRecomm_NoBias.initializeRatings("E:/code/NLF/train.txt",
				"E:/code/NLF/test.txt", "::");
		CommonRecomm_NoBias.lambda = 0.00001;
		//CommonRecomm_NoBias.lambda = 0;
		CommonRecomm_NoBias.trainingRound = 1000;
		CommonRecomm_NoBias.featureDimension =8;
		CommonRecomm_NoBias.delayCount = 50; 
		CommonRecomm_NoBias.initiStaticFeatures();
		/*{
			NLF testBRISMF = new NLF();
			testBRISMF.ifBi = false;
			testBRISMF.ifBu = false;
			testBRISMF.printNegativeFeature();
			testBRISMF.train();
		}*/
		{
			NLF testBRISMF = new NLF();
			testBRISMF.ifBi = true;
			testBRISMF.ifBu = true;
			testBRISMF.ifMu = true;
			testBRISMF.printNegativeFeature();
			testBRISMF.train();
			long endTime = System.currentTimeMillis();    
			System.out.println("Running time：" + (endTime - startTime) + "ms");    
		}

	}
}
