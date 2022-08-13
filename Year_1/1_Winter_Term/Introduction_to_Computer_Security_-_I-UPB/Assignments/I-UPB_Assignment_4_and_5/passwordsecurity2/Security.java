package passwordsecurity2;


import javax.crypto.SecretKeyFactory;
import javax.crypto.spec.PBEKeySpec;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.security.spec.InvalidKeySpecException;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import passwordsecurity2.Database.MyResult;
import org.passay.*;
import org.passay.dictionary.Dictionary;
import org.passay.dictionary.DictionaryBuilder;

public class Security {

    // Source: https://howtodoinjava.com/java/java-security/how-to-generate-secure-password-hash-md5-sha-pbkdf2-bcrypt-examples/

    protected static String hash(String password, byte[] salt) throws InvalidKeySpecException {
        try {
            int iterations = 1000;
            char[] chars = password.toCharArray();

            PBEKeySpec spec = new PBEKeySpec(chars, salt, iterations, 64 * 8);
            SecretKeyFactory skf = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA512");
            byte[] hashByteArray = skf.generateSecret(spec).getEncoded();

            // Convert Byte Array to String and return it
            return new String(hashByteArray, StandardCharsets.UTF_8);
        } catch (NoSuchAlgorithmException e) {
            e.printStackTrace();
        }

        return "";
    }

    protected static byte[] getSalt() throws NoSuchAlgorithmException {
        // We always need to use a SecureRandom to create good salts, and in Java,
        // the SecureRandom class supports the “SHA1PRNG” pseudo random number generator algorithm
        SecureRandom sr = SecureRandom.getInstance("SHA1PRNG");

        //Create array for salt
        byte[] salt = new byte[16];

        //Get a random salt
        sr.nextBytes(salt);

        return salt;
    }

    // Source: https://www.tutorialspoint.com/passay/passay_quick_guide.htm

    protected static MyResult isValid(String user_password) throws IOException {
        List<Rule> rules = new ArrayList<>();
        //Rule 1: Password length should be in between 10 and 99 characters
        rules.add(new LengthRule(10, 99));

        //Rule 2: No whitespace allowed
        rules.add(new WhitespaceRule());

        //Rule 3: At least three Upper-case characters
        rules.add(new CharacterRule(EnglishCharacterData.UpperCase, 3));

        //Rule 4: At least three Lower-case characters
        rules.add(new CharacterRule(EnglishCharacterData.LowerCase, 3));

        //Rule 5: At least three digits
        rules.add(new CharacterRule(EnglishCharacterData.Digit, 3));

        //Rule 6: The password should not be in the rockyou.txt file
        Dictionary wordListDictionary = new DictionaryBuilder().addFile("passwordsecurity2/rockyou.txt").build();
        rules.add(new DictionaryRule(wordListDictionary));

        //Display error messages from file if password does not obey all the rules
        Properties props = new Properties();
        props.load(new FileInputStream("passwordsecurity2/passay.properties"));
        MessageResolver resolver = new PropertiesMessageResolver(props);

        PasswordValidator validator = new PasswordValidator(resolver, rules);
        PasswordData password = new PasswordData(user_password);
        RuleResult result = validator.validate(password);

        if (result.isValid()) {
            return new MyResult(true, "Valid Password");
        } else {
            return new MyResult(false, String.join("\n", validator.getMessages(result)));
        }
    }
}
