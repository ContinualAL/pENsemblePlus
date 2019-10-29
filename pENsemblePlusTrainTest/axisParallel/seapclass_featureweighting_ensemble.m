
clc
nFolds=200;
ninput=3;
noutput=2;
decreasingfactor=0.5;
threshold=0.01;
confidenceinterval=0.001;
lambdaD=0.001;
lambdaW=0.008;
partial=0;
subset=2;
LR=0.2;
RF=0.01;
local=1;
s=0.05;
p=1;
kprune=0.3;
kfs=1.0;
vigilance=9.99*10^(-1);
type_feature_weighting=8;
RSMnew=0;
RSMdev=0;
CBtrain=[SEAtrainingdata SEAtrainingclass];
CBtest=[SEAtestingdata SEAtestingclass];
%CB=[CBtrain;CBtest];
[nData,nData1]=size(CBtrain);
%spambase=[spambasedatatrain;spambasedatatest]
[nDatatest,nDatatest1]=size(CBtest);
[creditcardoutput,pendigits_Data]=modify_dataset_zero_class(CBtrain);
[creditcardoutput1,pendigits_Data1]=modify_dataset_zero_class(CBtest);
[wineInputs1]=normal_class(CBtrain(:,1:end-1));
[wineInputs2]=normal_class(CBtest(:,1:end-1));
CBtrain=[wineInputs1 creditcardoutput];
CBtest=[wineInputs2 creditcardoutput1];
chunk_size=nData/nFolds;
chunk_size1=nDatatest/nFolds;

ensembleoutput=[];
inputpruning=1;
ensemblepruning1=0;
ensemblepruning2=1;
ensemblesize=[];

A1=[];
B=[];
C=[];
D=[];
E=[];
F=[];
l=0;
for i=1:chunk_size1:nDatatest
    l=l+1;
    if (i+chunk_size1-1) > nDatatest
        Data1 = CBtest(i:nDatatest,:);    %Tn = T(n:nTrainingData,:);
        %Block = size(Pn,1);             %%%% correct the block size
       % clear V;                        %%%% correct the first dimention of V 
    else
        Data1 = CBtest(i:(i+chunk_size1-1),:);   % Tn = T(n:(n+Block-1),:);
    end
    Data2(:,:,l)=Data1;
end
buffer=[];
counter=0;
ensemble=0;


for k=1:chunk_size:nData
    tic
counter=counter+1;  
  if (k+chunk_size-1) > nData
        Data = CBtrain(k:nData,:);    %Tn = T(n:nTrainingData,:);
        %Block = size(Pn,1);             %%%% correct the block size
       % clear V;                        %%%% correct the first dimention of V 
    else
        Data = CBtrain(k:(k+chunk_size-1),:);   % Tn = T(n:(n+Block-1),:);
  end
[r,q]=size(Data);
count_samples=0;
Selectedsamples=[];
 inputexpectation=mean(Data(:,1:ninput));
 inputvariance=var(Data(:,1:ninput));
 temporary=zeros(chunk_size,ninput);
 [upperbound,upperboundlocation]=max(Data(:,1:ninput));
 [lowerbound,lowerboundlocation]=min(Data(:,1:ninput));
 for iter=1:size(Data,1)
     for iter1=1:ninput
     temporary(iter,iter1)=Data(iter,iter1)-inputexpectation(iter1);
     end
 end
if ensemble==0
fix_the_model=size(Data,1);

paramet(1)=kprune;
paramet(2)=kfs;
paramet(3)=vigilance;

demo='n';
mode='c';
drift=2;


Data_fix=[Data;Data2(:,:,counter)];
[Weight,Center,Spread,rule,y,error,rules_significance,rules_novelty,datum_novelty,age,input_significance,population,time_index,born,time,classification_rate_testing,normalized_out,feature_weights,population_class_cluster,focalpoints,sigmapoints]=pclass_local_mod_improved_feature_weighting4(Data_fix,fix_the_model,paramet,demo,ninput,mode,drift,type_feature_weighting);
buffer=Data;
[v,vv]=size(Center);
network_parameters=v*subset+(subset)^(2)*v+(subset+1)*v*noutput;
network=struct('Center',Center,'Spread',Spread,'Weight',Weight,'ensemble_weight',1,'feature_weights',feature_weights,'network_parameters',network_parameters,'fuzzy_rule',v,'CR',classification_rate_testing,'focalpoints',focalpoints,'sigmapoints',sigmapoints,'population',population,'population_class_cluster',population_class_cluster);
ensemble=ensemble+1;
ensemblesize(1)=1;
for k3=1:noutput
error(:,k3,1)=0;
covariance(1,:,k3)=0;
covariance(:,1,k3)=0;
end
covariance_old=covariance;
else
    covariance_old=covariance;
  traceinputweight=[];  
  thres=0.9;
  storeoutput=[];
    for k1=1:size(Data,1)
                stream=Data(k1,:);
                        xek = [1, stream(1:ninput)]';
                  %      if inputpruning==1
                                       weight_input1=ones(1,size(Data,2));
         if partial==1
         z_t=random('Binomial',1,0.2);
        if z_t==1
        c_t=randperm(ninput);
        selectedinput=c_t(1:subset);
        weight_input1(~selectedinput)=0;
        stream=weight_input1.*stream;
            traceinputweight(k1,:)=weight_input1(1:ninput);
    else
         traceinputweight(k1,:)=ones(1,ninput);
        end
         end
       centeroverall=[];
       spreadoverall=[];
       weightoverall=[];
       populationoverall=[];
       populationclassoverall=[];
        for m=1:ensemble
                centeroverall=[centeroverall;network(m).Center];
                spreadoverall=[spreadoverall;network(m).Spread];
                weightoverall=[weightoverall;network(m).Weight];
                populationoverall=[populationoverall network(m).population];
                               for k2=1:size(network(m).Center,1)
                                   % populationoverall(end+k2)=network(m).population(k2);
                populationclassoverall(:,:,end+k2)=network(m).population_class_cluster(:,:,k2);
                               end
        end
[totalrule,ndimension]=size(centeroverall);
   weight_input=zeros(ndimension,totalrule);
   prior=zeros(1,totalrule);
   class_joint_probability=zeros(1,totalrule);
   volume=zeros(1,totalrule);
   likelihood=zeros(1,totalrule);
   if local==1     
   di=zeros(totalrule,1);
        
               for k2=1:totalrule
        dis=abs(stream(1:ninput)-centeroverall(k2,:));
        dis1=dis./spreadoverall(k2,:);
        di(k2)=exp(-0.5*dis1*dis');
        prior(k2)=populationoverall(k2)/sum(populationoverall);
        class_joint_probability(k2)=sum(populationclassoverall(:,:,k2))/sum(sum(populationclassoverall));
        volume(k2)=prod(spreadoverall(k2,:));
        likelihood(k2)=di(k2)/(2*(volume(k2))^(0.5)*2^(ninput));
               end
        fsig=di/sum(di); 
        numerator=zeros(noutput,totalrule);
        for k2=1:totalrule
        for k3=1:noutput
        numerator(k3,k2)=likelihood(k2)*class_joint_probability(k2)*prior(k2);
        end
        end
        inputprobability=zeros(1,noutput);
        for k4=1:noutput
        inputprobability(k4)=sum(numerator(k4,:))/sum(sum(numerator));
        end
    for k2=1:totalrule   
        Psik1((k2-1)*(ndimension+1)+1:k2*(ndimension+1),1) = fsig(k2)*xek;    
    end
   
    ysem=Psik1'*weightoverall;
    [maxout,classlabel]=max(ysem);
    A=sort(ysem,'descend');
    B=A(1)/(A(1)+A(2));
    outputprobability=min(max(B,0),1);
    clear Psik1
else
             weightperrule=zeros(ninput+1,noutput,totalrule);
       weightperrule(:)=weightoverall;
       di=zeros(totalrule,noutput);
       for k2=1:totalrule 
           for out=1:noutput
           di(k2,out)=xek'*weightperrule(:,out,k2);
           end
       end
       ysem=sum(di)/sum(sum(di));
       [maxout,classlabel]=max(ysem);

   clear weightperrule ysem
   end

        
if k1==size(Data,1)
            Zstat=mean(Data(:,1:ninput));
    cuttingpoint=0;
        for cut=1:size(Data,1)
        Xstat=mean(Data(1:cut,1:ninput));
        [Xupper,Xupper1]=max(Data(1:cut,1:ninput));
        [Xlower,Xlower1]=min(Data(1:cut,1:ninput));
        Xbound=(Xupper-Xlower)*sqrt(((r-cut)/(2*cut*(r))*reallog(1/confidenceinterval)));
        Zbound=(upperbound-lowerbound).*sqrt(((r-cut)/(2*cut*(r))*reallog(1/confidenceinterval)));
        if mean(Xbound+Xstat)>=mean(Zstat+Zbound) && cut<r
            cuttingpoint=cut;
              Ystat=mean(Data(cuttingpoint+1:end,1:ninput));
                      [Yupper,Yupper1]=max(Data(cuttingpoint+1:end,1:ninput));
        [Ylower,Ylower1]=min(Data(cuttingpoint+1:end,1:ninput));
         Ybound=(Yupper-Ylower).*sqrt(((r-cuttingpoint)/(2*cuttingpoint*(r-cuttingpoint)))*reallog(1/lambdaD));
          Ybound1=(Yupper-Ylower).*sqrt(((r-cuttingpoint)/(2*cuttingpoint*(r-cuttingpoint)))*reallog(1/lambdaW));
            break
       
        end
        end
if cuttingpoint==0
Ystat=Zstat;  
            Ybound=(upperbound-lowerbound).*sqrt(((r-cut)/(2*cut*(r))*reallog(1/lambdaD)));
            Ybound1=(upperbound-lowerbound).*sqrt(((r-cut)/(2*cut*(r))*reallog(1/lambdaW)));
end

end
   if (outputprobability>thres || max(inputprobability)>thres) && k1~=size(Data,1) && count_samples>5
      % thres=thres*(1-s);
       ensemble=size(network,1);
                                   output=zeros(1,noutput);
                            pruning_list=[];
        for m=1:ensemble
            weighted_stream=network(m).feature_weights.*stream(1:ninput);
            xek = [1, weighted_stream]';
        [nrule,ndimension]=size(network(m).Center);
        if local==1
        di=zeros(nrule,1);
        for k2=1:nrule
        dis=abs(weighted_stream-network(m).Center(k2,:));
        dis1=dis./network(m).Spread(k2,:);
        di(k2)=exp(-0.5*dis1*dis');
        end
        fsig=di/sum(di); 
    clear Psik2
        for k2=1:nrule      
        Psik2((k2-1)*(ndimension+1)+1:k2*(ndimension+1),1) = fsig(k2)*xek;    
        end
    ysem=Psik2'*network(m).Weight;

    [maxout,classlabel]=max(ysem);
    [maxout1,trueclasslabel]=max(stream(ndimension+1:end));
        else
            [nrule,ndimension]=size(network(m).Center);
             weightperrule=zeros(ninput+1,noutput,nrule);
       weightperrule(:)=network(m).Weight;
       di=zeros(nrule,noutput);
       for k2=1:nrule 
           for out=1:noutput
           di(k2,out)=xek'*weightperrule(:,out,k2);
           end
       end
       
       ysem=sum(di)/sum(sum(di));
       [maxout,classlabel]=max(ysem);
    [maxout1,trueclasslabel]=max(stream(ndimension+1:end)); 
        end
        for iter=1:noutput
        storeoutput(k1,iter,m)=ysem(iter);
        end
        end
   else   
  
       Selectedsamples=[Selectedsamples;stream];
       count_samples=count_samples+1;
       [maxout1,trueclasslabel]=max(stream(ndimension+1:end)); 
       if classlabel~=trueclasslabel
       weightperrule=zeros(ninput+1,noutput,totalrule);
       weightperrule(:)=weightoverall;
       for k2=1:totalrule
       for out=1:noutput
           if local==1
           weightperrule(:,out,k2)=weightperrule(:,out,k2)-LR*RF*weightperrule(:,out,k2)-LR.*RF.*fsig(k2).*xek;
           else
               weightperrule(:,out,k2)=weightperrule(:,out,k2)-LR*RF*weightperrule(:,out,k2)-LR.*RF.*xek;
           end
 
            if partial==1
        weightperrule(:,out,k2)=weightperrule(:,out,k2)*min(1,10/(norm(weightperrule(:,out,k2))));
   else
  weightperrule(:,out,k2)=weightperrule(:,out,k2)*min(1,1/(sqrt(0.01)*norm(weightperrule(:,out,k2))));
            end
       end
       end
        for k2=1:totalrule
        for j=1:ndimension 
                weight_input(j,k2)=sum(weightperrule(j,:,k2));
        end
        end
            weight_input_total=zeros(1,ndimension);
    for j=1:ndimension
    weight_input_total(j)=sum(weight_input(j,:));
    end
    [values,index]=sort(abs(weight_input_total),'descend');
    weight_input1=ones(1,size(stream,2));
    weight_input1(index(subset+1:end))=0;
    stream=stream.*weight_input1;
    traceinputweight(k1,:)=weight_input1(1:ninput);
       else
          traceinputweight(k1,:)=ones(1,ninput); 
       end
                   %     end
                            output=zeros(1,noutput);
                            pruning_list=[];
        for m=1:ensemble
            weighted_stream=network(m).feature_weights.*stream(1:ninput);
            xek = [1, weighted_stream]';
        [nrule,ndimension]=size(network(m).Center);
        if local==1
        di=zeros(nrule,1);
        for k2=1:nrule
        dis=abs(weighted_stream-network(m).Center(k2,:));
        dis1=dis./network(m).Spread(k2,:);
        di(k2)=exp(-0.5*dis1*dis');
        end
        fsig=di/sum(di); 
        clear Psik2
    for k2=1:nrule      
        Psik2((k2-1)*(ndimension+1)+1:k2*(ndimension+1),1) = fsig(k2)*xek;    
    end
    ysem=Psik2'*network(m).Weight;
    [maxout,classlabel]=max(ysem);
    [maxout1,trueclasslabel]=max(stream(ndimension+1:end));
        else
            [nrule,ndimension]=size(network(m).Center);
             weightperrule=zeros(ninput+1,noutput,nrule);
       weightperrule(:)=network(m).Weight;
       di=zeros(nrule,noutput);
       for k2=1:nrule 
           for out=1:noutput
           di(k2,out)=xek'*weightperrule(:,out,k2);
           end
       end
       
       ysem=sum(di)/sum(sum(di));
       [maxout,classlabel]=max(ysem);
    [maxout1,trueclasslabel]=max(stream(ndimension+1:end)); 
        end
        for iter=1:noutput
        storeoutput(k1,iter,m)=ysem(iter);
        end
    for out=ninput+1:size(stream,2)    
        error(count_samples,out-ninput,m) = ysem(out-ninput) - stream(out);     
    end
    Remp=zeros(1,noutput);

for out=1:noutput
Remp(out)=sumsqr(error(:,out,m))/k1;
end
    if classlabel~=trueclasslabel 
    network(m).ensemble_weight=network(m).ensemble_weight*decreasingfactor;
    else
      network(m).ensemble_weight=min(network(m).ensemble_weight*(2-decreasingfactor),1);  
    end
        output(classlabel)=output(classlabel)+network(m).ensemble_weight;
    clear Psik2
        end
               clear weightperrule ysem
        [maxout,ensemblelabel]=max(output);
        ensembleoutput(k1)=ensemblelabel;
        ensemblesize(k1)=ensemble;
        if mod(k1,p)==0
           
activation=0;
     if   k1==size(Data,1) 
            SSM=zeros(1,ensemble);
            a=zeros(1,ensemble);
            pruning_list=[];
            a=0;
           for m=1:ensemble
           a=a+network(m).ensemble_weight;
           end
         
        for m=1:ensemble
            
            network(m).ensemble_weight=network(m).ensemble_weight/a;     
 
             
        if  ensemblepruning1==1     && network(m).ensemble_weight<threshold && ensemble>1 && length(pruning_list)<size(network,1)
        pruning_list=[pruning_list m];
        end
        end
        if length(pruning_list)>=size(network,1) 
            pruning_list(end)=[];
        end
          if isempty(pruning_list)==false
                 network(pruning_list,:)=[];
        error(:,:,pruning_list)=[];
        ensemble=size(network,1);
        ensemblesize(k1)=ensemble;
                RSMnew(pruning_list)=[];
        RSMdev(pruning_list)=[];
        covariance(:,pruning_list,:)=[];
        covariance(pruning_list,:,:)=[];
        covariance_old=covariance;
        activation=1;
          end
     end

            pruning_list=[];
            %outputvar=zeros(noutput,ensemble);
            outputcovar=zeros(ensemble,ensemble,noutput);
            %outputmean=zeros(noutput,ensemble);
            if k1==size(Data,1) && ensemblepruning2==1 
                        for iter=1:ensemble
                        %    for iter1=1:ensemble
            %outputvar(iter1,iter)=var(storeoutput(:,iter1,iter));
            %outputmean(iter1,iter)=mean(storeoutput(:,iter1,iter));
             %               end
                        for iter1=1:ensemble
                            for iter2=1:noutput
                            temporary=cov(storeoutput(:,iter2,iter1),storeoutput(:,iter2,iter));
                        outputcovar(iter,iter1,iter2)=temporary(1,2);
                        covariance(iter,iter1,iter2)=(covariance_old(iter,iter1,iter2)*(counter-1)+(((counter-1)/counter)*outputcovar(iter,iter1,iter2)))/counter;
                            end
                        end
                        end
            end

             covariance_old=covariance;
                     if ensemble>1 && activation==0 && k1==size(Data,1) && ensemblepruning2==1 
            
              merged_list=[];
                for l=0:ensemble-2
        for hh=1:ensemble-l-1
             MCI=zeros(1,noutput);
            for o=1:noutput
            pearson=covariance(end-l,hh,o)/sqrt(covariance(end-l,end-l,o)*covariance(hh,hh,o));
            MCI(o)=(0.5*(covariance(hh,hh,o)+covariance(end-l,end-l,o))-sqrt((covariance(hh,hh,o)+covariance(end-l,end-l,o))^(2)-4*covariance(end-l,end-l,o)*covariance(hh,hh,o)*(1-pearson^(2))));
            end
       
                                           if max(abs(MCI))<0.1%(max(MCI)<0.1 & max(MCI)>0) & (max(MCI)>-0.1 & max(MCI)<0)
           if isempty(merged_list)
          merged_list(1,1)=ensemble-l;
          merged_list(1,2)=hh;
           else
                          No=find(merged_list(:,1:end-1)==ensemble-l);
            No1=find(merged_list(:,1:end-1)==hh);
            if isempty(No) && isempty(No1)
          merged_list(end+1,1)=ensemble-l;
          merged_list(end+1,2)=hh;
            end
           end
           break
                               end 
        end
                end
      
                                        del_list=[];
                                          RMSE=zeros(noutput,size(network,1));
for m=1:size(network,1)
for out=1:noutput
RMSE(out,m)=sumsqr(error(:,out,m))/k1;
end
end
Rselected=zeros(1,size(network,1));
for m=1:size(network,1)
Rselected(m)=mean(RMSE(:,m));
end
                    for i=1:size(merged_list,1)
                    No2=find(merged_list(i,:)==0);
                    if isempty(No2)
                                            if Rselected(merged_list(i,1))>Rselected(merged_list(i,2))
                      a=merged_list(i,1);
                      b=merged_list(i,2);
                      else
                        b=merged_list(i,1);
                      a=merged_list(i,2);    
                                            end
                                              del_list=[del_list b];                     
                    end
                    end
                       if isempty(del_list)==false
                 network(del_list,:)=[];
        error(:,:,del_list)=[];
        RSMnew(del_list)=[];
        RSMdev(del_list)=[];
        ensemble=size(network,1);
        ensemblesize(k1)=ensemble;
                covariance(:,pruning_list,:)=[];
        covariance(pruning_list,:,:)=[];
        covariance_old=covariance;
                       end
         
        end
        if k1==size(Data,1)


         if mean((abs(Xstat-Ystat)))>=mean(Ybound)
            %% drift
           
     fix_the_model=size(Selectedsamples,1);

paramet(1)=kprune;
paramet(2)=kfs;
paramet(3)=vigilance;


demo='n';
mode='c';
drift=2;


if isempty(buffer)
%Data_fix=Selectedsamples;
buffer=[Selectedsamples;Data2(:,:,counter)];
else
    buffer=[buffer;Selectedsamples;Data2(:,:,counter)];
end
[Weight,Center,Spread,rule,y,error,rules_significance,rules_novelty,datum_novelty,age,input_significance,population,time_index,born,time,classification_rate_testing,normalized_out,feature_weights,population_class_cluster,focalpoints,sigmapoints]=pclass_local_mod_improved_feature_weighting4(buffer,fix_the_model,paramet,demo,ninput,mode,drift,type_feature_weighting);
[v,vv]=size(Center);
network_parameters=v*subset+(subset)*v+(subset+1)*v*noutput;
network=[network; struct('Center',Center,'Spread',Spread,'Weight',Weight,'ensemble_weight',1,'feature_weights',feature_weights,'network_parameters',network_parameters,'fuzzy_rule',v,'CR',classification_rate_testing,'focalpoints',focalpoints,'sigmapoints',sigmapoints,'population',population,'population_class_cluster',population_class_cluster)];
ensemble=size(network,1);
ensemblesize(k1)=ensemble;
        RSMnew(ensemble)=0;
        RSMdev(ensemble)=0;
for k3=1:noutput
error(:,k3,ensemble)=0;
covariance(:,ensemble,k3)=0;
covariance(ensemble,:,k3)=0;
end
covariance_old=covariance;
buffer=[];
        elseif mean((abs(Xstat-Ystat)))>=mean(Ybound1) && mean((abs(Xstat-Ystat)))<mean(Ybound)
            %% Warning
            buffer=[buffer;Selectedsamples];
          
        else
            %%stable
              RMSE=zeros(noutput,size(network,1));
for m=1:size(network,1)
for out=1:noutput
RMSE(out,m)=sumsqr(error(:,out,m))/k1;
end
end
Rselected=zeros(1,size(network,1));
for m=1:size(network,1)
Rselected(m)=mean(RMSE(:,m));
end
[Rselected,index1]=min(Rselected);
  fix_the_model=size(Selectedsamples,1);


paramet(1)=kprune;
paramet(2)=kfs;
paramet(3)=vigilance;

buffer=[];

[Weight,Center,Spread,rule,y,error1,population,time,classification_rate_testing,feature_weights,population_class_cluster,focalpoints,sigmapoints]=pclass_updated(Selectedsamples,fix_the_model,paramet,demo,ninput,mode,drift,type_feature_weighting,network(index1));
[v,vv]=size(Center);
network_parameters=v*subset+(subset)*v+(subset+1)*v*noutput;
replacement=struct('Center',Center,'Spread',Spread,'Weight',Weight,'ensemble_weight',1,'feature_weights',feature_weights,'network_parameters',network_parameters,'fuzzy_rule',v,'CR',classification_rate_testing,'focalpoints',focalpoints,'sigmapoints',sigmapoints,'population',population,'population_class_cluster',population_class_cluster);
      network(index1)=replacement;
        end
    end
    end
    % % Start the model evolution (learning and prediction)
   end
    end
end

Datatest=Data2(:,:,counter);
ensembleoutputtest=zeros(size(Datatest,1),1);
individualoutputtest=zeros(size(Datatest,1),size(network,1));
misclassification=0;
outens=[];
for k1=1:size(Datatest,1)
    
    stream=Datatest(k1,:);
     output=zeros(1,noutput);

 for m=1:ensemble
              weighted_stream=network(m).feature_weights.*stream(1:ninput);
            xek = [1, weighted_stream]';
        [nrule,ndimension]=size(network(m).Center);
        if local==1
        di=zeros(nrule,1);
        for k2=1:nrule
        dis=(weighted_stream-network(m).Center(k2,:));
        dis1=dis./network(m).Spread(k2,:);
        di(k2)=exp(-0.5*dis1*dis');
        end
        fsig=di/sum(di); 
    for k2=1:nrule      
        Psik3((k2-1)*(ndimension+1)+1:k2*(ndimension+1),1) = fsig(k2)*xek;    
    end

    ysem=Psik3'*network(m).Weight;

    [maxout,classlabel]=max(ysem);
    individualoutputtest(k1,m)=classlabel;
     output(classlabel)=output(classlabel)+network(m).ensemble_weight;
     
     clear Psik3
        else
             [nrule,ndimension]=size(network(m).Center);
             weightperrule=zeros(ndimension+1,noutput,nrule);
             
       weightperrule(:)=network(m).Weight;
       di=zeros(nrule,noutput);
       for k2=1:nrule 
           for out=1:noutput
           di(k2,out)=xek'*weightperrule(:,out,k2);
           end
       end
        ysem=sum(di)/sum(sum(di));
       [maxout,classlabel]=max(ysem);
    individualoutputtest(k1,m)=classlabel;
     output(classlabel)=output(classlabel)+network(m).ensemble_weight;
        
        end
 end
 clear weightperrule ysem
 [maxout1,trueclasslabel]=max(stream(ndimension+1:end));
      [maxout,ensemblelabel]=max(output);
        ensembleoutputtest(k1)=ensemblelabel;
        if trueclasslabel==ensemblelabel
            misclassification=misclassification+1;
        end
        
end
totalrule=0;
totalparameters=0;
for m=1:size(network,1)
    totalrule=totalrule+network(m).fuzzy_rule;
    totalparameters=totalparameters+network(m).network_parameters;
end
time=toc;


A1(counter)=(misclassification)/size(Datatest,1);
C(counter)=totalrule;
D(counter)=totalparameters;
%

E(counter)=size(network,1);
if counter==1
F(counter)=size(Data,1);
else
F(counter)=count_samples;
end

H(counter)=time;

end


Brat=mean(A1);
Bdev=std(A1);
Crat=mean(C);
Cdev=std(C);
Drat=mean(D);
Ddev=std(D);
%
Erat=mean(E);
Edev=std(E);
%
Frat=mean(F);
Fdev=std(F);

Hrat=mean(H);
Hdev=std(H);


