function  [err, variance] = MAPE(actual,predicted)
    notNanIdxs = find(~isnan(actual) & ~isnan(predicted));
    actual = actual(notNanIdxs); predicted = predicted(notNanIdxs);
    err = 100*mean( abs((actual - predicted)./actual));
    variance = 100*std((actual - predicted)./actual)./(mean((actual - predicted)./actual));
end