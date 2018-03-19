import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA


class Data_Preprocessing:
    def __init__(self):
        self.df = pd.read_csv('covtype.csv')
        self.Original_df = self.df
        self.df = self.PCA_Reduction(self.df, ['Hillshade_9am','Hillshade_Noon','Hillshade_3pm'], 1, 'Hillshade')
        self.df = self.PCA_Reduction(self.df, ['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology'], 1, 'Distance_To_Hydrology')
        self.Result_From_OneHotExtraction = self.One_Hot_Extraction(self.df)
        #----------One Hot Data--------#
        self.Soil_Type_OneHot = self.Result_From_OneHotExtraction[0][0]
        self.Area_Type_OneHot = self.Result_From_OneHotExtraction[0][1]
        self.OneHot_SoilArea = np.array([item.tolist()+elem.tolist() for item, elem in zip(self.Soil_Type_OneHot, self.Area_Type_OneHot)])
        #------------------------------#
        self.df = self.One_Hot_Transfer(self.df, self.Result_From_OneHotExtraction)
        self.columns = list(self.df.columns)

    def PCA_Reduction(self, df, columns, components_left, New_name):
        PCA_Original = np.array(df[columns]).T
        pca = PCA(n_components = components_left)
        pca.fit(PCA_Original)
        for i in range(len(columns)):
            if i >= components_left:
                del df[columns[i]]
            else:
                PCA_New = pca.components_[i]
                df[columns[i]] = PCA_New
                index = list(df.columns).index(columns[i])
                df.rename(columns = {df.columns[index]: New_name+'_PCA_'+str(i)}, inplace=True)
        return df

    def One_Hot_Extraction(self, df):
        Area_Num = 4
        Soil_Type = 40
        Soil_Type = ['Soil_Type'+str(i+1) for i in range(Soil_Type)]
        Wilderness_Area = ['Wilderness_Area'+str(i+1) for i in range(Area_Num)]
        result = [np.array(df[Soil_Type]), np.array(df[Wilderness_Area])]
        for i in range(len(Soil_Type)):
            if i != 0:
                del df[Soil_Type[i]]
            else:
                index = list(df.columns).index(Soil_Type[i])
                df.rename(columns = {df.columns[index]: 'Soil_Type'}, inplace = True)
        for i in range(len(Wilderness_Area)):
            if i != 0:
                del df[Wilderness_Area[i]]
            else:
                index = list(df.columns).index(Wilderness_Area[i])
                df.rename(columns = {df.columns[index]: 'Wilderness_Area'}, inplace = True)
        return [result,df]

    def One_Hot_Transfer(self, df, Result_From_OneHotExtraction):
        temp_np = Result_From_OneHotExtraction[0]
        df = Result_From_OneHotExtraction[1]
        Final_Transfered_Data = []
        for item in temp_np:
            New_data = np.array([item[i].tolist().index(1) for i in range(len(item))])
            Final_Transfered_Data.append(New_data)
        df['Soil_Type'] = Final_Transfered_Data[0]
        df['Wilderness_Area'] = Final_Transfered_Data[1]
        return df

class Training_Model:
    def __init__(self, estimators):
        self.DP = Data_Preprocessing()
        self.estimators = estimators
        print self.DP.columns
        #---------Soil&Area_To_Coverage Training Input&Output-------------#
        self.X_Input_NonePre = np.array(self.DP.df[self.DP.columns[:-2]])
        self.X_Input = preprocessing.scale(np.array(self.DP.df[self.DP.columns[:-2]]))
        self.Y_SoilModel_Output = np.array(self.DP.df['Soil_Type'])
        self.X_Covreage = preprocessing.scale(np.array(self.DP.df[self.DP.columns[:-1]]))
        self.Y_Coverage = np.array(self.DP.df['Cover_Type'])
        #----------------Training-------------------#
        self.Training_SoilArea_To_Coverage(5)
        # rf = RandomForestClassifier(n_estimators=self.estimators)
        # self.five_fold(self.X_Input, self.Y_Coverage, rf, 5)
    # validation accuracy: 90%
    def Soil_Type_Training(self, X_Train, Y_Train, X_Test, Y_Test):
        rf = RandomForestClassifier(n_estimators=self.estimators)
        Model = rf.fit(X_Train, Y_Train)
        Y_Pre = rf.predict(X_Test)
        Model_Accuracy = accuracy_score(Y_Test, Y_Pre)
        return Y_Pre, Model_Accuracy, Model

    # Validation Accuracy is 96.5%
    def Training_FinalCover(self, X_Train, Y_Train, X_Test, Y_Test):
        rf = RandomForestClassifier(n_estimators=self.estimators)
        Model = rf.fit(X_Train, Y_Train)
        Y_Pre = rf.predict(X_Test)
        Model_Accuracy = accuracy_score(Y_Test, Y_Pre)
        return Y_Pre, Model_Accuracy, Model

    # Validation Accuracy: 95%
    def Soil_To_Coverage(self, X_input_tst, Soil_Pre, Y_Coverage_tst, Model):
        X = preprocessing.scale(np.array([item.tolist()+[j] for item, j in zip(X_input_tst, Soil_Pre)]))
        Y = Y_Coverage_tst
        Y_Pre = Model.predict(X)
        Acc = accuracy_score(Y, Y_Pre)
        return Y_Pre, Acc

    def Training_SoilArea_To_Coverage(self, n):
        print '-----------------Soil_To_Coverage Training Process---------------------'
        kf = KFold(n=len(self.X_Input), n_folds=n, shuffle=True)
        counter = 0
        soil_counter = 0
        cover_counter = 0
        combined_counter = 0
        for tr, tst in kf:
            counter = counter + 1
            print 'The '+str(counter)+'st Fold validation: '
            Soil_Pre, Soil_Acc, Soil_Model = self.Soil_Type_Training(self.X_Input[tr], self.Y_SoilModel_Output[tr], self.X_Input[tst], self.Y_SoilModel_Output[tst])
            Cover_Pre, Cover_Acc, Cover_Model = self.Training_FinalCover(self.X_Covreage[tr], self.Y_Coverage[tr], self.X_Covreage[tst], self.Y_Coverage[tst])
            Combined_Pre, Combined_Acc = self.Soil_To_Coverage(self.X_Input_NonePre[tst], Soil_Pre, self.Y_Coverage[tst], Cover_Model)
            soil_counter += Soil_Acc
            cover_counter += Cover_Acc
            combined_counter += Combined_Acc
            print 'Soil Accuracy: ' + str(Soil_Acc)
            print 'Coverage Accuracy: ' + str(Cover_Acc)
            print 'CombinedModel Accuracy: ' + str(Combined_Acc)
            #break
        print '-----------------Final Average Accuracy-----------------'
        print 'Average Accuracy of Soil Model: ' + str(soil_counter/counter)
        print 'Average Accuracy of Coverage Model: ' + str(cover_counter/counter)
        print 'Average Accuracy of Combined Model: ' + str(combined_counter/counter)

    def five_fold(self, X, Y, model, n):
        kf = KFold(n=len(Y), n_folds=n, shuffle=True)
        cv = 0
        average = []
        for tr, tst in kf:
            tr_features = X[tr]
            tr_target = Y[tr]
            tst_features = X[tst]
            tst_target = Y[tst]
            model.fit(tr_features, tr_target)
            tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
            tst_accuracy = np.mean(model.predict(tst_features) == tst_target)
            print "%d Fold Train Accuracy:%f, Test Accuracy:%f" % (cv+1, tr_accuracy, tst_accuracy)
            average.append(tst_accuracy)
            cv += 1
        ave = sum(np.array(average))/cv
        print 'average accuracy of the model: '+ str(ave)


Training_Model(200) # the parameter: The number of Decision Trees in Random Forest
