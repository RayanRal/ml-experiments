import pandas as pnd
from skimage.io import imread


image = imread('parrots_4.jpg')

# input data
# FullDescription  LocationNormalized  ContractTime  SalaryNormalized
inputData = pnd.read_csv('close_prices.csv', index_col=False)
testData = pnd.read_csv('djia_index.csv', index_col=False)
testData = testData.copy().drop(inputData.columns[3], axis=1)
# testData = testData.loc[0,:]
# print("testData row one: {}").format(testData)

# filling missing values
inputData['LocationNormalized'].fillna('nan', inplace=True)
inputData['ContractTime'].fillna('nan', inplace=True)

testData['LocationNormalized'].fillna('nan', inplace=True)
testData['ContractTime'].fillna('nan', inplace=True)

#text preprocessing
inputData['FullDescription'] = inputData['FullDescription'].str.lower()
inputData['LocationNormalized'] = inputData['LocationNormalized']#.str.lower()
inputData['ContractTime'] = inputData['ContractTime']#.str.lower()
inputData['FullDescription'] = inputData['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)


testData['FullDescription'] = inputData['FullDescription'].str.lower()
testData['LocationNormalized'] = inputData['LocationNormalized']#.str.lower()
testData['ContractTime'] = inputData['ContractTime']#.str.lower()
testData['FullDescription'] = inputData['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)


# TfidfVectorizer
vect = TfidfVectorizer(min_df=5)
fullDescriptionTransformed = TfidfVectorizer.fit_transform(vect, raw_documents=inputData['FullDescription'])
fullDescriptionTransformedTest = TfidfVectorizer.transform(vect, raw_documents=testData['FullDescription'])

# one-hot encoding
enc = DictVectorizer()
otherTransformed = enc.fit_transform(inputData[['LocationNormalized', 'ContractTime']].to_dict('records'))
otherTransformedTest = enc.transform(testData[['LocationNormalized', 'ContractTime']].to_dict('records'))

trainFeatures = hstack([fullDescriptionTransformed, otherTransformed])
testFeatures = hstack([fullDescriptionTransformedTest, otherTransformedTest])

# print("trainFeatures: {}").format(trainFeatures.get_shape)
trainTarget = inputData.copy().drop(inputData.columns[0:3], axis=1)

model = Ridge(alpha=1)#, solver='lsqr', fit_intercept=False)
fitted = model.fit(X=trainFeatures, y=trainTarget)
# print("testFeatures: {}").format(testFeatures.get_shape)
result = fitted.predict(testFeatures)

print("result: {}").format(result)