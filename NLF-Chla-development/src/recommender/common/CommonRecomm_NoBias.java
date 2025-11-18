package recommender.common;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.StringTokenizer;

public abstract class CommonRecomm_NoBias{

	public boolean ifBu = false;

	public boolean ifBi = false;

	public boolean ifMu = false;

	public double minMAE = 100;

	public double minRMSE = 100;

	public int minRound = 0;

	public static int delayCount = 10;

	public static double maxRating = 5;

	public static double minRating = 0;


	public double Mu = 0;
	
	public double userBias[];

	public double minUserBias[];

	public static double catchedUserBias[];

	public double itemBias[];

	public double minItemBias[];

	public static double catchedItemBias[];

	public double[][] userFeatureArray;

	public double[][] minUserFeatureArray;

	public static double[][] catchedUserFeatureArray;

	public double[][] itemFeatureArray;

	public double[][] minItemFeatureArray;

	public static double[][] catchedItemFeatureArray;

	public static double[][] userFactorUp, userFactorDown, itemFactorUp,
			itemFactorDown;

	public static double[] userBiasUp, userBiasDown, itemBiasUp, itemBiasDown;

	public static int featureDimension = 100;

	public static int trainingRound = 500;
	

	public static double userFeatureInitMax = 0.004;

	public static double userFeatureInitScale = 0.004;

	public static double itemFeatureInitMax = 0.004;

	public static double itemFeatureInitScale = 0.004;
	

	public static int mappingScale = 1000;


	public static double eta = 0.01;

	public static double lambda = 0.005;
	public static double lambda_b = 0.005;
	


	public static ArrayList<RTuple> trainData = null;

	public static ArrayList<RTuple> testData = null;

	public abstract void train() throws IOException;

	public CommonRecomm_NoBias() throws NumberFormatException, IOException {
		this.initInstanceFeatures();
	}

	public static void initiStaticFeatures() {

		catchedItemFeatureArray = new double[maxItemID + 1][featureDimension];
		catchedUserFeatureArray = new double[maxUserID + 1][featureDimension];
		catchedUserBias = new double[maxUserID + 1];
		catchedItemBias = new double[maxItemID + 1];

		Random random = new Random(System.currentTimeMillis());
		for (int i = 1; i <= maxUserID; i++) {
			int tempUB = random.nextInt(mappingScale);
	
			catchedUserBias[i] = userFeatureInitMax - userFeatureInitScale
					* tempUB / mappingScale;    
			for (int j = 0; j < featureDimension; j++) {
				int temp = random.nextInt(mappingScale);
				catchedUserFeatureArray[i][j] = userFeatureInitMax
						- userFeatureInitScale * temp / mappingScale;
			}
		}
		for (int i = 1; i <= maxItemID; i++) {
			int tempIB = random.nextInt(mappingScale);
			catchedItemBias[i] = itemFeatureInitMax - itemFeatureInitScale
					* tempIB / mappingScale;
			for (int j = 0; j < featureDimension; j++) {
				int temp = random.nextInt(mappingScale);
				catchedItemFeatureArray[i][j] = itemFeatureInitMax
						- itemFeatureInitScale * temp / mappingScale;
			}
		}
		initNMFAuxArrays();
	}

	public static void initNMFAuxArrays() {

		userFactorUp = new double[maxUserID + 1][featureDimension];
		userFactorDown = new double[maxUserID + 1][featureDimension];
		itemFactorUp = new double[maxItemID + 1][featureDimension];
		itemFactorDown = new double[maxItemID + 1][featureDimension];

		userBiasUp = new double[maxUserID + 1];
		userBiasDown = new double[maxUserID + 1];
		itemBiasUp = new double[maxItemID + 1];
		itemBiasDown = new double[maxItemID + 1];

	}

	public static void resetNMFAuxArrays() {
		for (int i = 1; i <= maxUserID; i++) {
			userBiasUp[i] = 0;
			userBiasDown[i] = 0;
			for (int j = 0; j < featureDimension; j++) {
				userFactorUp[i][j] = 0;
				userFactorDown[i][j] = 0;
			}
		}
		for (int i = 1; i <= maxItemID; i++) {
			itemBiasUp[i] = 0;
			itemBiasDown[i] = 0;
			for (int j = 0; j < featureDimension; j++) {
				itemFactorUp[i][j] = 0;
				itemFactorDown[i][j] = 0;
			}
		}
	}

	public void initInstanceFeatures() {
		userBias = new double[maxUserID + 1];
		itemBias = new double[maxItemID + 1];
		minUserBias = new double[maxUserID + 1];
		minItemBias = new double[maxItemID + 1];
		userFeatureArray = new double[maxUserID + 1][featureDimension];
		itemFeatureArray = new double[maxItemID + 1][featureDimension];
		minUserFeatureArray = new double[maxUserID + 1][featureDimension];
		minItemFeatureArray = new double[maxItemID + 1][featureDimension];
		for (int i = 1; i <= maxUserID; i++) {
			userBias[i] = catchedUserBias[i];
			for (int j = 0; j < featureDimension; j++) {
				userFeatureArray[i][j] = catchedUserFeatureArray[i][j];//把上面的随机值附进去
			}
		}
		for (int i = 1; i <= maxItemID; i++) {
			itemBias[i] = catchedItemBias[i];
			for (int j = 0; j < featureDimension; j++) {
				itemFeatureArray[i][j] = catchedItemFeatureArray[i][j];
			}
		}
	}

	public void cacheMinFeatures() {
		for (int i = 1; i <= maxUserID; i++) {
			minUserBias[i] = userBias[i];
			for (int j = 0; j < featureDimension; j++) {
				minUserFeatureArray[i][j] = userFeatureArray[i][j];
			}
		}
		for (int i = 1; i <= maxItemID; i++) {
			minItemBias[i] = itemBias[i];
			for (int j = 0; j < featureDimension; j++) {
				minItemFeatureArray[i][j] = itemFeatureArray[i][j];
			}
		}
	}

	public static int maxItemID = 0, maxUserID = 0;

	public static void initializeRatings(String trainFileName,
			String testFileName, String separator)
			throws NumberFormatException, IOException {
		initTrainData(trainFileName, separator);
		initTestData(testFileName, separator);
		initRatingSetSize();
	}

	public static void initTrainData(String fileName, String separator)
			throws NumberFormatException, IOException {
		trainData = new ArrayList<RTuple>();
		File personSource = new File(fileName);
		BufferedReader in = new BufferedReader(new FileReader(personSource));
		//int mm=0;
		String tempVoting;
		while (((tempVoting = in.readLine()) != null)) {
			StringTokenizer st = new StringTokenizer(tempVoting, separator);//StringTokenizer对tempVoting进行解析，分隔符为separator
			String personID = null;
			if (st.hasMoreTokens())
				personID = st.nextToken();
			String movieID = null;
			if (st.hasMoreTokens())
				movieID = st.nextToken();
			String personRating = null;
			if (st.hasMoreTokens())
				personRating = st.nextToken();
			int iUserID = Integer.valueOf(personID);
			int iItemID = Integer.valueOf(movieID);
			maxUserID = (maxUserID > iUserID) ? maxUserID : iUserID;
			maxItemID = (maxItemID > iItemID) ? maxItemID : iItemID;
			double dRating = Double.valueOf(personRating);

			RTuple temp = new RTuple();
			temp.iUserID = iUserID;
			temp.iItemID = iItemID;
			temp.dRating = dRating;
			trainData.add(temp);
		}
	}

	public static double userRSetSize[], itemRSetSize[];

	public static void initRatingSetSize() {
		userRSetSize = new double[maxUserID + 1];
		itemRSetSize = new double[maxItemID + 1];

		for (int i = 0; i <= maxUserID; i++)
			userRSetSize[i] = 0;
		for (int i = 0; i <= maxItemID; i++)
			itemRSetSize[i] = 0;
		for (RTuple tempRating : trainData) {
			userRSetSize[tempRating.iUserID] += 1;
			itemRSetSize[tempRating.iItemID] += 1;
		}
	}

	public static void initTestData(String fileName, String separator)
			throws NumberFormatException, IOException {
		testData = new ArrayList<RTuple>();
		File personSource = new File(fileName);
		BufferedReader in = new BufferedReader(new FileReader(personSource));
		String tempVoting;
		while (((tempVoting = in.readLine()) != null)) {
			StringTokenizer st = new StringTokenizer(tempVoting, separator);
			String personID = null;
			if (st.hasMoreTokens()){
				personID = st.nextToken();  
			}		
			String movieID = null;
			if (st.hasMoreTokens())
				movieID = st.nextToken();
			String personRating = null;
			if (st.hasMoreTokens())
				personRating = st.nextToken();
			int iUserID = Integer.valueOf(personID);
			int iItemID = Integer.valueOf(movieID);
			maxUserID = (maxUserID > iUserID) ? maxUserID : iUserID;
			maxItemID = (maxItemID > iItemID) ? maxItemID : iItemID;

			double dRating = Double.valueOf(personRating);

			RTuple temp = new RTuple();
			temp.iUserID = iUserID;
			temp.iItemID = iItemID;
			temp.dRating = dRating;
			testData.add(temp);
		}

	}

	public double getPrediciton(int userID, int itemID) {
		return dotMultiply(userFeatureArray[userID], itemFeatureArray[itemID]);
	}

	public static double dotMultiply(double[] x, double[] y) {
		double sum = 0;
		for (int i = 0; i < x.length; i++) {
			sum += x[i] * y[i];
		}
		return sum;
	}

	public static void vectorAdd(double[] first, double[] second,
			double[] result) {
		for (int i = 0; i < first.length; i++) {
			result[i] = first[i] + second[i];
		}
	}

	public static void vectorSub(double[] first, double[] second,
			double[] result) {
		for (int i = 0; i < first.length; i++) {
			result[i] = first[i] - second[i];
		}
	}

	public static void vectorMutiply(double[] vector, double time,
			double[] result) {
		for (int i = 0; i < vector.length; i++) {
			result[i] = vector[i] * time;
		}
	}

	public static double[] initZeroVector() {
		double[] result = new double[featureDimension];
		for (int i = 0; i < featureDimension; i++)
			result[i] = 0;
		return result;
	}

	public void printNegativeFeature() {

		System.out
				.println("************************** negative user bias:*****************************");
		for (int userID = 1; userID <= maxUserID; userID++) {
			if (userBias[userID] < 0)
				System.out.println(userBias[userID]);
		}

		System.out
				.println("************************** negative user feature:**************************");
		for (int userID = 1; userID <= maxUserID; userID++) {
			for (int tempF = 0; tempF < featureDimension; tempF++) {
				if (userFeatureArray[userID][tempF] < 0)
					System.out.println(userFeatureArray[userID][tempF]);
			}
		}

		System.out
				.println("************************** negative item bias:*****************************");
		for (int itemID = 1; itemID <= maxItemID; itemID++) {
			if (itemBias[itemID] < 0)
				System.out.println(itemBias[itemID]);
		}

		System.out
				.println("************************** negative item feature:**************************");
		for (int itemID = 1; itemID <= maxItemID; itemID++) {
			for (int tempF = 0; tempF < featureDimension; tempF++) {
				if (itemFeatureArray[itemID][tempF] < 0)
					System.out.println(itemFeatureArray[itemID][tempF]);
			}
		}
		System.out
				.println("***************************************************************************");

	}

	public double getMu() {
		double tempMu = 0, tempCount = 0;
		for (RTuple tempRating : trainData) {
			tempMu += tempRating.dRating;
			tempCount++;
		}
		tempMu = tempMu / tempCount;
		Mu = tempMu;
		
		return tempMu;
	}

	public double getMinPrediction(int userID, int itemID, boolean ifMu,
			boolean ifBi, boolean ifBu) {
		double ratingHat = 0;
		ratingHat += dotMultiply(minUserFeatureArray[userID],
				minItemFeatureArray[itemID]);
		if (ifMu)
			ratingHat += Mu;
		if (ifBi)
			ratingHat += minItemBias[itemID];
		if (ifBu)
			ratingHat += minUserBias[userID];
		return ratingHat;
	}


	public double getMinPrediction(int userID, int itemID) {
		return this.getMinPrediction(userID, itemID, this.ifMu, this.ifBi,
				this.ifBu);
	}

	public double getLocPrediction(int userID, int itemID, boolean ifMu,
			boolean ifBi, boolean ifBu) {
		double ratingHat = 0;
		ratingHat += dotMultiply(userFeatureArray[userID],
				itemFeatureArray[itemID]);
		if (ifMu)
			ratingHat += Mu;
		if (ifBi)
			ratingHat += itemBias[itemID];
		if (ifBu)
			ratingHat += userBias[userID];
		return ratingHat;
	}

	public double getLocPrediction(int userID, int itemID) {
		return this.getLocPrediction(userID, itemID, this.ifMu, this.ifBi,
				this.ifBu);
	}

	public void outputMinFeatures() throws IOException {
	    //System.out.println("outputMinFeatures");
		FileWriter fw_bias = new FileWriter("E:/code/NLF/Out/"
				+ new File(this.getClass().getName().trim() + "_"
						+ System.currentTimeMillis() + "chla_bias.csv"));
		
		FileWriter fw_item = new FileWriter("E:/code/NLF/Out/"
                + new File(this.getClass().getName().trim() + "_"
                        + System.currentTimeMillis()+ "chla_item.txt"));
		
		FileWriter fw_user = new FileWriter("E:/code/NLF/Out/"
                + new File(this.getClass().getName().trim() + "_"
                        + System.currentTimeMillis()+ "chla_user.txt"));
		
		FileWriter output = new FileWriter("E:/code/NLF/Out/"
				+ new File(this.getClass().getName().trim() + "_"
						+ System.currentTimeMillis()+ "chla_output.txt"));
		fw_bias.write("Mu\n");
		fw_bias.write(this.Mu + "\n");
		fw_bias.flush();
		fw_bias.write("UserBias\n");
		for (int i = 1; i <= maxUserID; i++) {
		    fw_bias.write(this.minUserBias[i] + "\n");
		    fw_bias.flush();
		}
		fw_bias.write("ItemBias\n");
		for (int i = 1; i <= maxItemID; i++) {
		    fw_bias.write(this.minItemBias[i] + "\n");
		    fw_bias.flush();
		}
		fw_bias.close();
		fw_user.write("UserFeature\n");
		for (int i = 1; i <= maxUserID; i++) {
			String temp = "";
			for (int j = 0; j < featureDimension - 1; j++) {
				temp += this.minUserFeatureArray[i][j] + ",";
			}
			temp += this.minUserFeatureArray[i][featureDimension - 1];
			fw_user.write(temp + "\n");
			fw_user.flush();
		}
		fw_user.close();
		fw_item.write("ItemFeature\n");
		for (int i = 1; i <= maxItemID; i++) {
			String temp = "";
			for (int j = 0; j < featureDimension - 1; j++) {
				temp += this.minItemFeatureArray[i][j] + ",";
			}
			temp += this.minItemFeatureArray[i][featureDimension - 1];
			fw_item.write(temp + "\n");
			fw_item.flush();
		}
		fw_item.close();
				String sep = "::";				
				for (int n = 1; n <= maxUserID; n++) {
					for (int m = 1; m <= maxItemID; m++) {
						double prediction = getMinPrediction(n, m);
						double mu = getMu();
						if (m != maxItemID)
							output.write(Double.toString(prediction) + sep);
						else
						    output.write(Double.toString(prediction));
						   }
					output.write("\n");
						}
				output.close();		
	}

	public void readMinFeaturesFromFile(String FileName) throws IOException {
		BufferedReader in = new BufferedReader(new FileReader(
				new File(FileName)));
		String currentLine;
		int currentType = 0;
		int itemBiasCount = 0, userBiasCount = 0, userFeatureCount = 0, itemFeatureCount = 0;
		while (((currentLine = in.readLine()) != null)) {
			if (currentLine.equals("Mu")) {
				currentType = 1;
				continue;
			}
			if (currentLine.equals("UserBias")) {
				currentType = 2;
				continue;
			}
			if (currentLine.equals("ItemBias")) {
				currentType = 3;
				continue;
			}
			if (currentLine.equals("UserFeature")) {
				currentType = 4;
				continue;
			}
			if (currentLine.equals("ItemFeature")) {
				currentType = 5;
				continue;
			}
			switch (currentType) {
			case 1:
				this.Mu = Double.valueOf(currentLine);
				break;
			case 2:
				userBiasCount++;
				this.minUserBias[userBiasCount] = Double.valueOf(currentLine);
				break;
			case 3:
				itemBiasCount++;
				this.minItemBias[itemBiasCount] = Double.valueOf(currentLine);
				break;

			case 4: {
				StringTokenizer st = new StringTokenizer(currentLine, "::");
				int featureCount = 0;
				userFeatureCount++;
				while (st.hasMoreTokens()) {
					this.minUserFeatureArray[userFeatureCount][featureCount] = Double
							.valueOf(st.nextToken());
					featureCount++;
				}
			}
				break;
			case 5: {
				StringTokenizer st = new StringTokenizer(currentLine, "::");
				int featureCount = 0;
				itemFeatureCount++;
				while (st.hasMoreTokens()) {
					this.minItemFeatureArray[itemFeatureCount][featureCount] = Double
							.valueOf(st.nextToken());
					featureCount++;
				}
			}
				break;
			default:
				break;
			}
		}
	}
}
