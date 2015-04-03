function Gradient_Descent_regularized_new(train_X,train_Y,test_X,test_Y)

[train_X_n,train_X_f]=size(train_X);
[test_X_n,test_X_f]=size(test_X);

lambda=0.05;
step=0.01;

T=50;
train_re_CE=zeros(T,1);     %regularized corss-entropy error
test_re_CE=zeros(T,1);     %regularized corss-entropy error


w=zeros(train_X_f,T+1);
b=[0.1,zeros(1,T)];

for t=1:T
    
    sum_w=zeros(train_X_f,1);
    sum_b=0;
    
    for n=1:train_X_n
        
        sigmoid=1/(1+exp(-w(:,t)'*train_X(n,:)'-b(t)));
        
        sum_b=sum_b+sigmoid-train_Y(n);
        sum_w=sum_w+(sigmoid-train_Y(n))*train_X(n,:)';
        
    end
    
    w(:,t+1)=w(:,t)-step*sum_w-2*lambda*step*w(:,t);
    b(t+1)=b(t)-step*sum_b;
    
    % CE for train data
    
    for n=1:train_X_n
        
        sigmoid=1/(1+exp(-w(:,t+1)'*train_X(n,:)'-b(t+1)));
        if sigmoid<10^(-16)
            sigmoid=10^(-16);
        elseif sigmoid>(1-10^(-16))
            sigmoid=1-10^(-16);
        end
        train_re_CE(t)=train_re_CE(t)-train_Y(n)*log(sigmoid)-(1-train_Y(n))*log(1-sigmoid);
        
    end
    
    train_re_CE(t)=train_re_CE(t)+lambda*norm(w(:,t+1))^2;
    
    % CE for test data
    
    for n=1:test_X_n
        
        sigmoid=1/(1+exp(-(w(:,t+1)'*test_X(n,:)'+b(t+1))));
        if sigmoid<10^(-16)
            sigmoid=10^(-16);
        elseif sigmoid>(1-10^(-16))
            sigmoid=1-10^(-16);
        end
        test_re_CE(t)=test_re_CE(t)-test_Y(n)*log(sigmoid)-(1-test_Y(n))*log(1-sigmoid);
        
    end
    
    test_re_CE(t)=test_re_CE(t)+lambda*norm(w(:,t+1))^2;
    
end
